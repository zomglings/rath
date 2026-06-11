/**
 * request_human_edit: a client-side tool that hands a file to the human's
 * editor and blocks until they save and quit, then returns what they wrote.
 *
 * This is the human-in-the-loop primitive for co-authoring documents: the
 * agent drafts text, opens it in the human's editor, and reads back the
 * result. The editor is the human's own ($VISUAL/$EDITOR, with a detected
 * fallback), so they edit with the tool they already know.
 *
 * The tool is frontend-agnostic. Spawning an editor needs sole ownership of
 * the terminal, which the TUI does not have while running; the caller passes a
 * `suspendTerminal` hook that suspends its UI for the duration (the plain REPL
 * needs none and uses the pass-through default).
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, isAbsolute, join } from "node:path";
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { createPatch } from "diff";
import { Type } from "typebox";
import { isOnPath } from "../which.js";

/**
 * Editors that fork and return immediately unless told to wait for the file to
 * be closed. For these we append a wait flag so spawnSync actually blocks
 * until the human is done (the same reason git wants `core.editor = "code
 * --wait"`). Keys are matched against the resolved command's basename.
 */
const GUI_EDITORS_NEEDING_WAIT = new Set([
  "code",
  "code-insiders",
  "codium",
  "vscodium",
  "cursor",
  "windsurf",
  "subl",
  "sublime_text",
  "mate",
]);

// Fallback editors to probe on PATH when neither $VISUAL nor $EDITOR is set,
// in preference order. code first (the common case here), then the terminal
// editors most likely to be installed.
const FALLBACK_EDITORS =
  process.platform === "win32" ? ["code", "notepad"] : ["code", "vim", "emacs", "nano"];

/** The basename of `cmd` without a Windows executable extension, lowercased. */
function commandKey(cmd: string): string {
  return basename(cmd)
    .toLowerCase()
    .replace(/\.(exe|cmd|bat|com)$/, "");
}

/**
 * Resolve the editor command (argv) to run. Order: explicit override, then
 * $VISUAL, then $EDITOR (split on spaces so "code --wait" works), then the
 * first FALLBACK_EDITOR found on PATH. Throws if nothing is available. A GUI
 * editor that would otherwise return immediately gets a `--wait` appended.
 */
export function resolveEditorCommand(override?: string): string[] {
  const raw = (override ?? process.env.VISUAL ?? process.env.EDITOR ?? "").trim();
  let argv: string[];
  if (raw.length > 0) {
    argv = raw.split(/\s+/);
  } else {
    const found = FALLBACK_EDITORS.find((e) => isOnPath(e));
    if (!found) {
      throw new Error(
        `No editor found: set $VISUAL or $EDITOR (tried ${FALLBACK_EDITORS.join(", ")})`,
      );
    }
    argv = [found];
  }
  const [cmd, ...rest] = argv;
  if (cmd === undefined) {
    throw new Error("Empty editor command");
  }
  const hasWaitFlag = rest.some((a) => a === "-w" || a === "--wait");
  if (GUI_EDITORS_NEEDING_WAIT.has(commandKey(cmd)) && !hasWaitFlag) {
    argv = [...argv, "--wait"];
  }
  return argv;
}

/**
 * Open `file` in `editorArgv`, blocking until the editor exits. Returns true on
 * a clean exit, false otherwise. stdio is inherited so the editor owns the
 * terminal; the caller must have suspended any competing UI first.
 */
function openEditor(editorArgv: string[], file: string): boolean {
  const [cmd, ...rest] = editorArgv;
  if (cmd === undefined) {
    return false;
  }
  // On Windows the resolved editor is often a .cmd shim, which spawnSync can
  // only run through a shell; elsewhere we exec it directly.
  const useShell = process.platform === "win32";
  const result = spawnSync(cmd, [...rest, file], { stdio: "inherit", shell: useShell });
  return !(result.error || (typeof result.status === "number" && result.status !== 0));
}

export interface RequestHumanEditDetails {
  /** Absolute path of the edited file (a temp file when no path was given). */
  path: string;
  /** Whether the human changed the file. */
  changed: boolean;
  /** Unified diff of the human's changes ("" when unchanged). */
  diff: string;
  /** True when the file was a tool-created temp file rather than a given path. */
  temp: boolean;
}

export interface RequestHumanEditOptions {
  /** Base directory for resolving a relative `path`. Defaults to process.cwd(). */
  cwd?: string;
  /** Editor command override; defaults to $VISUAL/$EDITOR then a PATH probe. */
  editor?: string;
  /**
   * Run `fn` while the caller's UI is suspended so the editor owns the
   * terminal, returning fn's result. Defaults to calling fn directly (correct
   * for the plain REPL, which holds no UI mid-turn).
   */
  suspendTerminal?: <T>(fn: () => T) => T;
}

const PARAMETERS = Type.Object({
  content: Type.Optional(
    Type.String({
      description:
        "Initial contents to put in the file before opening the editor (a draft for the human to revise). Omit to open the existing file at `path`, or an empty document when no path is given.",
    }),
  ),
  path: Type.Optional(
    Type.String({
      description:
        "File to open. Relative paths resolve against the working directory. Omit to use a throwaway temp file whose path is returned alongside the result.",
    }),
  ),
});

/**
 * Build the request_human_edit AgentTool. See module docs. The returned tool
 * blocks the agent loop while the human edits; on save+quit it returns the
 * final file contents plus a unified diff of their changes.
 */
export function createRequestHumanEditTool(
  options: RequestHumanEditOptions = {},
): AgentTool<typeof PARAMETERS, RequestHumanEditDetails> {
  const cwd = options.cwd ?? process.cwd();
  const suspend = options.suspendTerminal ?? (<T>(fn: () => T): T => fn());
  return {
    name: "request_human_edit",
    label: "Request human edit",
    description:
      "Open a file in the human's text editor and wait for them to edit, save, and quit. " +
      "Use this to collaborate on a document: optionally seed it with a draft via `content`, " +
      "the human revises it directly, and you receive back the final contents and a diff of " +
      "their changes. Blocks until the editor is closed.",
    parameters: PARAMETERS,
    execute: async (_toolCallId, params): Promise<AgentToolResult<RequestHumanEditDetails>> => {
      const editorArgv = resolveEditorCommand(options.editor);

      const given = params.path?.trim();
      const isTemp = !given;
      const targetPath = given
        ? isAbsolute(given)
          ? given
          : join(cwd, given)
        : join(mkdtempSync(join(tmpdir(), "rath-edit-")), "document.txt");

      // Seed the file: explicit content wins; otherwise ensure the file exists
      // so the editor opens cleanly (a fresh temp file or a brand-new path).
      if (params.content !== undefined) {
        writeFileSync(targetPath, params.content);
      } else if (!existsSync(targetPath)) {
        writeFileSync(targetPath, "");
      }

      const before = readFileSync(targetPath, "utf8");
      const ran = suspend(() => openEditor(editorArgv, targetPath));
      if (!ran) {
        throw new Error(`Editor exited abnormally: ${editorArgv.join(" ")} ${targetPath}`);
      }
      const after = readFileSync(targetPath, "utf8");
      const changed = after !== before;
      const name = basename(targetPath);
      const diff = changed ? createPatch(name, before, after, "before", "after") : "";

      const header = changed
        ? `The human edited ${targetPath}.`
        : `The human made no changes to ${targetPath}.`;
      const sections = [header];
      if (changed) {
        sections.push(`Diff:\n${diff}`);
      }
      sections.push(`Final contents of ${name}:\n${after}`);

      return {
        content: [{ type: "text", text: sections.join("\n\n") }],
        details: { path: targetPath, changed, diff, temp: isTemp },
      };
    },
  };
}
