/**
 * `rath run`: a generic agent loop with nothing implicit.
 *
 * No skill discovery, no context-file walking (AGENTS.md is never read). The
 * model sees exactly what the flags specify: the system prompt, the loaded
 * context, and the user's prompts. The only thing taken from the environment
 * is the provider API key. Client-side tools are the one convenience default:
 * --tools defaults to all of them (pass --tools none to disable, or a list to
 * choose), since rath development inside `rath run` wants them on hand.
 *
 * Built on pi-agent-core's Agent (the stateful loop wrapper: transcript,
 * lifecycle events, tool execution, model switching via state.model) and
 * pi-ai's provider registry; the openai-native provider is registered, so
 * hosted web search is available (on by default, --no-web-search disables
 * it). Client-side tools are loaded on demand from
 * @earendil-works/pi-coding-agent. Frontends: a plain readline REPL
 * (default, also used for -p one-shots) and a pi-tui interface (-T/--tui)
 * rendered with Pi's own interactive components. The TUI's dependencies
 * (pi-tui and pi-coding-agent's UI) are imported inside runTui, so they only
 * load when --tui is used; the type-only imports below are erased at compile
 * time and cost nothing.
 */
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as readline from "node:readline/promises";
import {
  Agent,
  type AgentTool,
  type BeforeToolCallContext,
  type BeforeToolCallResult,
} from "@earendil-works/pi-agent-core";
import {
  type Api,
  type AssistantMessage,
  getModel,
  getModels,
  getProviders,
  getSupportedThinkingLevels,
  type KnownProvider,
  type Message,
  type Model,
  type SimpleStreamOptions,
  streamSimple,
} from "@earendil-works/pi-ai";
import type * as PiCodingAgent from "@earendil-works/pi-coding-agent";
import type * as PiTui from "@earendil-works/pi-tui";
import { type Command, fullName, helpText } from "../command.js";
import { clearDefaultModel, loadPreferences, setDefaultModel } from "../config.js";
import {
  applyCitationTrailer,
  contentBlocks,
  flattenHostedContent,
  isHostedToolCall,
  isRenderedCitations,
  OPENAI_NATIVE_API,
  OPENROUTER_NATIVE_API,
  openaiNativeModel,
  openrouterNativeModel,
  registerOpenAINative,
  registerOpenRouterNative,
  stripRenderedCitations,
  uniqueUrlCitations,
} from "../index.js";
import { createRequestHumanEditTool } from "../tools/request-human-edit.js";
import { isOnPath } from "../which.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

// dim() wraps status text, which the plain frontend writes to stderr; gate the
// escape codes on stderr's TTY status so piping stdout keeps the dimming and
// redirecting stderr to a file does not capture raw escapes.
function dim(text: string): string {
  return process.stderr.isTTY ? `${DIM}${text}${RESET}` : text;
}

export const REASONING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"] as const;
export type ReasoningLevel = (typeof REASONING_LEVELS)[number];

export const TOOL_NAMES = [
  "read",
  "bash",
  "edit",
  "write",
  "grep",
  "find",
  "ls",
  "request_human_edit",
] as const;
export type ToolName = (typeof TOOL_NAMES)[number];

// Client-side tools sourced from @earendil-works/pi-coding-agent (everything
// except rath's own request_human_edit).
type PiToolName = Exclude<ToolName, "request_human_edit">;

// Interaction modes for `rath run`:
// - "go": full speed, no inspection. Stream everything, never block, tools run
//   immediately. This is the default and must stay byte-for-byte the current
//   behavior — no paging, the gating hook is a pass-through.
// - "slow": take time. Long output is paged (a pager in the plain REPL, a
//   scrollable overlay in the TUI), and every tool call is gated behind a
//   per-call confirmation before it runs.
export const MODES = ["go", "slow"] as const;
export type Mode = (typeof MODES)[number];

// The default for the (DB-stored) default model: used when neither -m nor a
// pinned default model (preferences, set via /config default-model) supplies
// one. Hence the doubled name — it is the default of the default.
export const DEFAULT_DEFAULT_MODEL = `${OPENAI_NATIVE_API}/gpt-5.5`;

export interface RunFlags {
  model: string;
  prompt?: string;
  systemPrompt: string;
  reasoning: ReasoningLevel;
  webSearch: boolean;
  tools: ToolName[];
  tui: boolean;
  mode: Mode;
  loadPath?: string;
  savePath?: string;
}

// Paging threshold: output longer than this many lines is paged in slow mode.
// Slash-command output and tool results are usually well-bounded, so a fixed
// line count is simpler and more predictable than chasing terminal height.
export const LONG_OUTPUT_LINES = 24;

/** True when `text` is long enough to page in slow mode. */
export function isLongOutput(text: string): boolean {
  // Count lines: a trailing newline does not add an empty line to read.
  let lines = 1;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "\n" && i !== text.length - 1) {
      lines++;
    }
  }
  return lines > LONG_OUTPUT_LINES;
}

/**
 * Load the requested client-side tools, preserving the requested order. The
 * pi-coding-agent tools are imported lazily (the package is large) and only
 * when at least one is requested. request_human_edit is rath's own tool; it
 * needs `suspendTerminal` so it can take the terminal from the active frontend
 * while the editor runs (the TUI passes a tui.stop/start wrapper; the plain
 * REPL needs none).
 */
export async function loadTools(
  names: ToolName[],
  cwd: string,
  suspendTerminal?: <T>(fn: () => T) => T,
): Promise<AgentTool[]> {
  const piNames = names.filter((n): n is PiToolName => n !== "request_human_edit");
  let piFactories: Record<PiToolName, (cwd: string) => AgentTool> | undefined;
  if (piNames.length > 0) {
    const pi = await import("@earendil-works/pi-coding-agent");
    // Typed as AgentTool so pi-coding-agent API drift fails compilation here.
    piFactories = {
      read: pi.createReadTool,
      bash: pi.createBashTool,
      edit: pi.createEditTool,
      write: pi.createWriteTool,
      grep: pi.createGrepTool,
      find: pi.createFindTool,
      ls: pi.createLsTool,
    };
  }
  return names.map((name) =>
    name === "request_human_edit"
      ? (createRequestHumanEditTool({ cwd, suspendTerminal }) as AgentTool)
      : piFactories![name](cwd),
  );
}

/**
 * Lets a tool that must own the terminal (request_human_edit's editor) suspend
 * whatever UI the active frontend is running. The plain REPL holds no UI
 * mid-turn, so its controller is a pass-through; the TUI swaps in a
 * tui.stop()/start() wrapper once it is up.
 */
interface TerminalController {
  suspend: <T>(fn: () => T) => T;
}

export function resolveModel(spec: string): Model<Api> {
  const slash = spec.indexOf("/");
  if (slash <= 0 || slash === spec.length - 1) {
    throw new Error(`Model must be <provider>/<model-id> (got: ${spec})`);
  }
  const provider = spec.slice(0, slash);
  const modelId = spec.slice(slash + 1);
  if (provider === OPENAI_NATIVE_API) {
    return openaiNativeModel(modelId);
  }
  if (provider === OPENROUTER_NATIVE_API) {
    return openrouterNativeModel(modelId) as Model<Api>;
  }
  const model = getModel(provider as KnownProvider, modelId as never) as Model<Api> | undefined;
  if (!model) {
    throw new Error(`Unknown model: ${spec}`);
  }
  return model;
}

/** All selectable model specs, native providers first. */
export function listModels(filter?: string): string[] {
  const specs: string[] = [];
  for (const model of getModels("openai")) {
    specs.push(`${OPENAI_NATIVE_API}/${model.id}`);
  }
  for (const model of getModels("openrouter")) {
    specs.push(`${OPENROUTER_NATIVE_API}/${model.id}`);
  }
  for (const provider of getProviders()) {
    for (const model of getModels(provider)) {
      specs.push(`${provider}/${model.id}`);
    }
  }
  return filter ? specs.filter((s) => s.includes(filter)) : specs;
}

export interface SerializedContext {
  systemPrompt?: string;
  messages: Message[];
}

export function loadContext(path: string): SerializedContext {
  const parsed = JSON.parse(readFileSync(path, "utf8")) as SerializedContext;
  if (!Array.isArray(parsed.messages)) {
    throw new Error(`${path} does not contain a serialized context (messages array missing)`);
  }
  return parsed;
}

export function saveContext(agent: Agent, path: string): void {
  const messages = agent.state.messages.filter(
    (m): m is Message =>
      m.role === "user" ||
      m.role === "toolResult" ||
      // Skip failed turns so they do not accumulate as empty bubbles across
      // saved sessions or replay as malformed items on --load.
      (m.role === "assistant" && m.stopReason !== "error" && m.stopReason !== "aborted"),
  );
  writeFileSync(
    path,
    `${JSON.stringify({ systemPrompt: agent.state.systemPrompt, messages }, null, 2)}\n`,
  );
}

/** Current configuration as key/value pairs, for the /config display. */
export function sessionInfo(agent: Agent, flags: RunFlags): [string, string][] {
  const tools = agent.state.tools.map((t) => t.name);
  const pinnedDefault = loadPreferences().defaultModel;
  return [
    ["model (session)", flags.model],
    [
      "default model",
      pinnedDefault ?? `built-in: ${DEFAULT_DEFAULT_MODEL} (/config default-model to pin)`,
    ],
    ["mode", flags.mode === "slow" ? "slow (paging + tool confirmation)" : "go (full speed)"],
    ["reasoning", String(agent.state.thinkingLevel)],
    ["web search", flags.webSearch ? "on (hosted, openai-native / openrouter-native only)" : "off"],
    ["tools (active)", tools.length > 0 ? tools.join(", ") : "none"],
    ["tools (available)", TOOL_NAMES.join(", ")],
    ["skills", "none (rath run loads no skills or context files)"],
    ["system prompt", "/sys to view or set"],
    ["messages in context", String(agent.state.messages.length)],
    ...(flags.loadPath ? [["loaded from", flags.loadPath] as [string, string]] : []),
    ["save on exit", flags.savePath ?? "off"],
  ];
}

export interface SlashResult {
  output?: string;
  isError?: boolean;
  exit?: boolean;
}

/**
 * Handle an in-session slash command. Returns undefined when the input is
 * not a slash command (i.e. it is a prompt for the model).
 */
export async function handleSlashCommand(
  input: string,
  agent: Agent,
  flags: RunFlags,
  terminal: TerminalController,
): Promise<SlashResult | undefined> {
  if (!input.startsWith("/")) {
    return undefined;
  }
  const [command = "", ...rest] = input.split(/\s+/);
  const arg = rest.join(" ").trim();
  // Most settings take effect next turn; the running turn already snapshotted.
  const deferred = agent.state.isStreaming ? " (applies after the current turn)" : "";
  switch (command) {
    case "/exit":
      return { exit: true };
    case "/config": {
      // Subcommands manage persisted preferences; bare /config shows everything.
      const [sub = "", ...subRest] = arg.split(/\s+/).filter((s) => s.length > 0);
      if (sub === "default-model") {
        const spec = subRest.join(" ").trim();
        if (spec.length === 0) {
          const pinned = loadPreferences().defaultModel;
          return {
            output: pinned
              ? `default model: ${pinned}`
              : `default model: (none pinned; built-in ${DEFAULT_DEFAULT_MODEL})`,
          };
        }
        if (spec === "none" || spec === "clear") {
          clearDefaultModel();
          return { output: `default model cleared (built-in ${DEFAULT_DEFAULT_MODEL})` };
        }
        try {
          resolveModel(spec);
        } catch (error) {
          return { output: error instanceof Error ? error.message : String(error), isError: true };
        }
        setDefaultModel(spec);
        return { output: `default model pinned: ${spec} (applies to new sessions)` };
      }
      if (sub.length > 0) {
        return {
          output: `Unknown /config subcommand: ${sub} (use /config or /config default-model [spec|none])`,
          isError: true,
        };
      }
      return {
        output: sessionInfo(agent, flags)
          .map(([key, value]) => `${key}: ${value}`)
          .join("\n"),
      };
    }
    case "/sys": {
      if (arg.length === 0) {
        const prompt = agent.state.systemPrompt;
        return { output: prompt.length > 0 ? prompt : "(empty)" };
      }
      agent.state.systemPrompt = arg;
      flags.systemPrompt = arg;
      return { output: `system prompt set (${arg.length} chars)${deferred}` };
    }
    case "/websearch": {
      if (arg.length === 0) {
        return { output: `web search: ${flags.webSearch ? "on" : "off"}` };
      }
      if (arg !== "on" && arg !== "off") {
        return { output: "Usage: /websearch [on|off]", isError: true };
      }
      flags.webSearch = arg === "on";
      return {
        output: `web search: ${arg} (openai-native / openrouter-native only)${deferred}`,
      };
    }
    case "/tools": {
      if (arg.length === 0) {
        const names = agent.state.tools.map((t) => t.name);
        return { output: `tools: ${names.length > 0 ? names.join(", ") : "none"}` };
      }
      if (arg === "none") {
        agent.state.tools = [];
        flags.tools = [];
        return { output: `tools: none${deferred}` };
      }
      const requested = arg
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
      const invalid = requested.filter((n) => !TOOL_NAMES.includes(n as ToolName));
      if (invalid.length > 0) {
        return {
          output: `Unknown tool(s): ${invalid.join(", ")} (use ${TOOL_NAMES.join(", ")})`,
          isError: true,
        };
      }
      const names = requested as ToolName[];
      try {
        agent.state.tools = await loadTools(names, process.cwd(), (fn) => terminal.suspend(fn));
        flags.tools = names;
        return { output: `tools: ${names.join(", ")}${deferred}` };
      } catch (error) {
        return { output: error instanceof Error ? error.message : String(error), isError: true };
      }
    }
    case "/save": {
      const path = arg.length > 0 ? arg : flags.savePath;
      if (!path) {
        return { output: "Usage: /save <path> (also sets the save-on-exit path)", isError: true };
      }
      try {
        saveContext(agent, path);
        flags.savePath = path;
        return { output: `saved to ${path}` };
      } catch (error) {
        return { output: error instanceof Error ? error.message : String(error), isError: true };
      }
    }
    case "/model": {
      if (arg.length === 0) {
        return { output: `model: ${flags.model}` };
      }
      try {
        agent.state.model = resolveModel(arg);
        flags.model = arg;
        return {
          output: `model: ${arg}${deferred} (/config default-model to pin across sessions)`,
        };
      } catch (error) {
        return { output: error instanceof Error ? error.message : String(error), isError: true };
      }
    }
    case "/lsmodels": {
      const specs = listModels(arg.length > 0 ? arg : undefined);
      const footer = `${specs.length} model${specs.length === 1 ? "" : "s"}${arg ? ` matching "${arg}"` : ""}`;
      return { output: specs.length > 0 ? `${specs.join("\n")}\n${footer}` : footer };
    }
    case "/reasoning": {
      const supported = getSupportedThinkingLevels(agent.state.model as Model<Api>);
      if (arg.length === 0) {
        return {
          output: `reasoning: ${agent.state.thinkingLevel} (supported by ${flags.model}: ${supported.join(", ")})`,
        };
      }
      if (!REASONING_LEVELS.includes(arg as ReasoningLevel)) {
        return {
          output: `Invalid reasoning level: ${arg} (use ${REASONING_LEVELS.join(", ")})`,
          isError: true,
        };
      }
      agent.state.thinkingLevel = arg as ReasoningLevel;
      flags.reasoning = arg as ReasoningLevel;
      // "off" is honored as-is (reasoning disabled), not clamped up to a
      // supported level — only positive levels outside the set are clamped.
      const clampNote =
        arg === "off" || supported.includes(arg as ReasoningLevel)
          ? ""
          : ` (outside ${flags.model}'s supported levels — the native provider clamps it)`;
      return { output: `reasoning: ${arg}${clampNote}${deferred}` };
    }
    case "/mode": {
      if (arg.length === 0) {
        return { output: `mode: ${flags.mode}` };
      }
      if (arg !== "go" && arg !== "slow") {
        return { output: "Usage: /mode [go|slow]", isError: true };
      }
      flags.mode = arg;
      return { output: `mode: ${arg}` };
    }
    case "/go":
      flags.mode = "go";
      return { output: "mode: go (full speed)" };
    case "/slow":
      flags.mode = "slow";
      return { output: "mode: slow (paging + tool confirmation)" };
    default:
      return {
        output:
          `Unknown command: ${command} (commands: /config [default-model [spec|none]], ` +
          "/sys [text], /model [spec], /lsmodels [filter], /reasoning [level], " +
          "/websearch [on|off], /tools [names|none], /mode [go|slow], /go, /slow, " +
          "/save [path], /exit)",
        isError: true,
      };
  }
}

interface Renderer {
  /** True when the most recent run produced an error message. */
  hadError: () => boolean;
}

/**
 * Wire plain rendering onto the agent's event stream. The model's answer text
 * goes to stdout (the deliverable, so `-p > file` captures just that);
 * thinking, hosted-tool markers, tool-execution lines, and errors go to
 * stderr as status.
 *
 * When `pageReplies` is set (the interactive REPL, never `-p`), slow mode
 * buffers the answer instead of streaming it and pages the finished reply
 * through $PAGER so it can be read deliberately; short replies and go mode
 * stream/print as usual.
 */
function attachRenderer(agent: Agent, flags: RunFlags, pageReplies = false): Renderer {
  // Track which stream has an unterminated line so the next block starts fresh.
  let openStream: NodeJS.WriteStream | null = null;
  const write = (stream: NodeJS.WriteStream, text: string) => {
    stream.write(text);
    openStream = text.endsWith("\n") ? null : stream;
  };
  const ensureNewline = () => {
    if (openStream) {
      openStream.write("\n");
      openStream = null;
    }
  };
  const seenHostedCalls = new Set<string>();
  let errored = false;
  // In slow mode (interactive only) the thinking and answer are not streamed
  // live; the finished turn is paged at message_end so it can be read at
  // leisure. Tool-call markers still stream as progress.
  const paging = () => pageReplies && flags.mode === "slow";
  // Print a (possibly partial) turn with the same stream split as live
  // streaming: answer to stdout, thinking and the sources trailer to stderr.
  // Used when slow mode suppressed the live stream but cannot page (short
  // output, no TTY) or the turn was interrupted.
  const printTurnSplit = (message: AssistantMessage) => {
    for (const block of contentBlocks(message)) {
      if (block.type === "thinking") {
        const t = block.thinking.trim();
        if (t.length > 0) {
          write(process.stderr, `${dim(t)}\n`);
        }
      } else if (block.type === "text") {
        const t = block.text.trim();
        if (t.length === 0) {
          continue;
        }
        if (isRenderedCitations(block)) {
          write(process.stderr, `${dim(t)}\n`);
        } else {
          write(process.stdout, `${t}\n`);
        }
      }
    }
  };

  agent.subscribe((event) => {
    if (event.type === "agent_start") {
      errored = false;
    } else if (event.type === "message_update") {
      const e = event.assistantMessageEvent;
      if (e.type === "text_delta") {
        if (!paging()) {
          write(process.stdout, e.delta);
        }
      } else if (e.type === "thinking_delta") {
        if (!paging()) {
          write(process.stderr, dim(e.delta));
        }
      } else if (e.type === "text_end" || e.type === "thinking_end") {
        ensureNewline();
      }
      // Hosted tool calls emit no assistant-message events; they appear as
      // blocks on the partial message. Announce each one once.
      if (event.message.role === "assistant") {
        for (const block of contentBlocks(event.message)) {
          if (isHostedToolCall(block) && !seenHostedCalls.has(block.id)) {
            seenHostedCalls.add(block.id);
            ensureNewline();
            write(process.stderr, `${dim(`[${block.toolName}]`)}\n`);
          }
        }
      }
    } else if (event.type === "tool_execution_start") {
      ensureNewline();
      const args = JSON.stringify(event.args);
      write(
        process.stderr,
        `${dim(`[${event.toolName}: ${args.length > 120 ? `${args.slice(0, 120)}…` : args}]`)}\n`,
      );
    } else if (event.type === "message_end") {
      const message = event.message;
      if (message.role === "assistant") {
        ensureNewline();
        if (message.stopReason === "error" || message.stopReason === "aborted") {
          // Slow mode suppressed the live stream; surface whatever partial
          // content was generated before the interrupt/error rather than
          // discarding it.
          if (paging()) {
            printTurnSplit(message);
          }
          write(
            process.stderr,
            `${message.stopReason}: ${message.errorMessage ?? "interrupted"}\n`,
          );
          // A user-initiated interrupt is not a failure; only a genuine error
          // makes the session exit non-zero.
          if (message.stopReason === "error") {
            errored = true;
          }
          return;
        }
        const trailer = applyCitationTrailer(message);
        if (paging()) {
          // Page the finished turn — thinking, answer, and sources — through
          // $PAGER. If it cannot page (short output, or stdout is not a TTY),
          // print with the normal stream split so thinking does not land on
          // stdout.
          if (!pagePlain(buildPagedTurnText(message), flags)) {
            printTurnSplit(message);
          }
        } else if (trailer) {
          write(process.stderr, `${dim(trailer)}\n`);
        }
      }
    }
  });
  return { hadError: () => errored };
}

/** Short, single-line preview of tool arguments for a confirmation prompt. */
export function toolArgsPreview(args: unknown, max = 120): string {
  let text: string;
  try {
    text = JSON.stringify(args) ?? String(args);
  } catch {
    text = String(args);
  }
  // Collapse whitespace so the preview stays on one line.
  text = text.replace(/\s+/g, " ");
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

/**
 * Build a beforeToolCall hook for the Agent. In go mode the hook is a
 * pass-through, so go mode keeps the current "tools run immediately" behavior
 * exactly. In slow mode it asks `confirm` before every tool call and blocks
 * the call when the user declines; a frontend that cannot confirm (non-TTY
 * plain REPL, or -p one-shot) denies rather than passing through.
 *
 * The hook reads flags.mode on every call, so /slow mid-session starts gating
 * and /go stops it — no agent rebuild needed.
 *
 * This is also the security mitigation for prompt injection: with hosted web
 * search on, a malicious page can try to steer the model into running tools.
 * Slow mode forces a human to approve each tool call before it executes.
 */
export function makeBeforeToolCall(
  flags: RunFlags,
  confirm: (toolName: string, argsPreview: string, signal?: AbortSignal) => Promise<boolean>,
): (ctx: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined> {
  return async (
    ctx: BeforeToolCallContext,
    signal?: AbortSignal,
  ): Promise<BeforeToolCallResult | undefined> => {
    if (flags.mode !== "slow") {
      return undefined; // go mode: pass-through, no gating.
    }
    const toolName = ctx.toolCall.name;
    // An interrupt fired while the confirmation is pending counts as a deny so
    // the tool does not run after the user aborted the turn.
    if (signal?.aborted) {
      return { block: true, reason: `Tool call to ${toolName} denied (interrupted).` };
    }
    const approved = await confirm(toolName, toolArgsPreview(ctx.args), signal);
    if (approved) {
      return undefined;
    }
    return { block: true, reason: `Tool call to ${toolName} denied by user (slow mode).` };
  };
}

/**
 * Build the readable text of a finished assistant turn for the pager: thinking
 * (labeled), the answer, and the rendered Sources trailer, in content order.
 * Hosted/function tool-call blocks are omitted (their results are paged
 * separately); call after applyCitationTrailer so the trailer is included.
 */
export function buildPagedTurnText(message: AssistantMessage): string {
  const parts: string[] = [];
  for (const block of contentBlocks(message)) {
    if (block.type === "thinking") {
      const t = block.thinking.trim();
      if (t.length > 0) {
        parts.push(`Thinking:\n${t}`);
      }
    } else if (block.type === "text") {
      const t = block.text.trim();
      if (t.length > 0) {
        parts.push(t);
      }
    }
  }
  return parts.join("\n\n");
}

/**
 * The pager command (argv array) to use: $PAGER split on spaces if set,
 * otherwise the first detected default among `less -R` and `more`. Returns []
 * when nothing usable is found, which makes runExternalPager a no-op so the
 * caller renders inline.
 */
export function resolvePagerCommand(): string[] {
  const pagerEnv = (process.env.PAGER ?? "").trim();
  if (pagerEnv.length > 0) {
    // Split on spaces so PAGER="less -R" works (the common shell convention).
    return pagerEnv.split(/\s+/);
  }
  const defaults = [["less", "-R"], ["more"]];
  return defaults.find((argv) => argv[0] !== undefined && isOnPath(argv[0])) ?? [];
}

/**
 * Run the external pager ($PAGER, default `less -R`) on `text`, full-screen,
 * with vim/arrow/PgUp-PgDn navigation and search — i.e. real less behavior.
 * Returns true when the pager ran, false when no usable pager was found or it
 * exited abnormally (the caller should then render the content itself).
 *
 * The pager command comes from $PAGER if set; otherwise the first of `less -R`
 * and `more` whose executable is actually on PATH (so we degrade to whatever
 * the platform has rather than hard-failing on systems without less). See
 * resolvePagerCommand.
 *
 * The text is written to a temp file and the pager is run as `pager <file>`
 * with inherited stdio so it owns the terminal (drawing to the screen, reading
 * the keyboard). Piping on the pager's stdin instead would make less detect a
 * non-TTY stdin and dump like cat. The synchronous `spawnSync` blocks the
 * event loop for the pager's lifetime; the temp file is removed afterward. The
 * caller must own the terminal first (the plain REPL is between reads; the TUI
 * suspends itself with tui.stop()).
 */
export function runExternalPager(text: string): boolean {
  const parts = resolvePagerCommand();
  const [cmd, ...rest] = parts;
  if (!cmd) {
    return false;
  }
  let dir: string | undefined;
  try {
    dir = mkdtempSync(join(tmpdir(), "rath-pager-"));
    const file = join(dir, "output.txt");
    writeFileSync(file, text.endsWith("\n") ? text : `${text}\n`);
    const result = spawnSync(cmd, [...rest, file], { stdio: "inherit" });
    if (result.error || (typeof result.status === "number" && result.status !== 0)) {
      return false;
    }
    return true;
  } catch {
    return false;
  } finally {
    if (dir) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
}

/** Page `text` in the plain REPL when in slow mode and the output is long. */
export function pagePlain(text: string, flags: RunFlags): boolean {
  if (flags.mode !== "slow" || !isLongOutput(text)) {
    return false;
  }
  if (!process.stdout.isTTY || !process.stdin.isTTY) {
    return false;
  }
  return runExternalPager(text);
}

export const runCommand: Command = {
  name: "run",
  summary: "Run a generic agent loop",
  description:
    "Starts an agent loop with nothing implicit: no skill discovery, no\n" +
    "context-file walking. The model sees exactly what the flags specify. The\n" +
    "provider API key (e.g. OPENAI_API_KEY) is the only input taken from the\n" +
    "environment. Tools are the one convenience default: --tools enables all\n" +
    "client-side tools when omitted (--tools none to disable, or a list).\n" +
    "\n" +
    "Without --prompt, reads prompts interactively (plain REPL, or pi-tui\n" +
    "with -T/--tui). Every startup setting is also settable in-session:\n" +
    "  /config                show the configuration\n" +
    "  /config default-model [spec|none]  pin or clear the default model\n" +
    "  /sys [text]            show or set the system prompt\n" +
    "  /model [provider/id]   show or switch the session model\n" +
    "  /lsmodels [filter]     list available models\n" +
    "  /reasoning [level]     show or set the reasoning level\n" +
    "  /websearch [on|off]    show or toggle hosted web search\n" +
    "  /tools [names|none]    show or set client-side tools\n" +
    "  /mode [go|slow]        show or switch the interaction mode\n" +
    "  /go, /slow             switch mode (go: full speed; slow: page + confirm)\n" +
    "  /save [path]           write the context now and save there on exit\n" +
    "  /exit                  quit\n" +
    "With --prompt, runs one prompt and exits (0 on success).",
  flags: [
    {
      long: "model",
      short: "m",
      takesValue: true,
      description: "Model as <provider>/<model-id> (default: openai-native/gpt-5-mini)",
    },
    {
      long: "prompt",
      short: "p",
      takesValue: true,
      description: "Run a single prompt non-interactively and exit",
    },
    {
      long: "tui",
      short: "T",
      takesValue: false,
      description: "Interactive pi-tui interface instead of the plain REPL",
    },
    {
      long: "system-prompt",
      takesValue: true,
      description: "System prompt text (default: a one-line generic assistant prompt)",
    },
    {
      long: "system-prompt-file",
      takesValue: true,
      description: "Read the system prompt from a file",
    },
    {
      long: "reasoning",
      takesValue: true,
      description: "Reasoning effort: off, minimal, low, medium, high, xhigh (default: low)",
    },
    {
      long: "mode",
      takesValue: true,
      description:
        "Interaction mode: go (full speed, default) or slow (page long output " +
        "and confirm each tool call before it runs)",
    },
    {
      long: "no-web-search",
      takesValue: false,
      description: "Disable hosted web search (openai-native / openrouter-native)",
    },
    {
      long: "tools",
      takesValue: true,
      repeatable: true,
      description:
        "Client-side tools to enable (comma-separated or repeated): read, bash, edit, write, " +
        "grep, find, ls, request_human_edit. Omit to enable all of them; pass --tools none to " +
        "disable. They run with your privileges in the current directory.",
    },
    {
      long: "load",
      takesValue: true,
      description: "Load a previously saved context (JSON) before starting",
    },
    {
      long: "save",
      takesValue: true,
      description: "Save the context (JSON) to this path on exit",
    },
  ],
  async run(prefix, argv) {
    const flags: RunFlags = {
      model: DEFAULT_DEFAULT_MODEL,
      systemPrompt: "You are a helpful assistant.",
      reasoning: "low",
      webSearch: true,
      tools: [],
      tui: false,
      mode: "go",
    };
    let systemPromptFile: string | undefined;
    let systemPromptExplicit = false;
    // Tools default to all when --tools is omitted; an explicit --tools (even
    // "none") opts out of that default.
    let toolsExplicit = false;
    // -m wins; otherwise the last-used model from preferences becomes the
    // default, falling back to the built-in default below.
    let modelExplicit = false;
    for (let i = 0; i < argv.length; i++) {
      const token = argv[i]!;
      const value = (): string => {
        const v = argv[++i];
        if (v === undefined) {
          throw new Error(`Option ${token} requires a value`);
        }
        return v;
      };
      try {
        // Help is honored only as a standalone token, so a flag value of
        // "-h"/"--help" (e.g. --prompt --help) is taken literally.
        if (token === "-h" || token === "--help") {
          process.stdout.write(`${helpText(this, prefix)}\n`);
          return 0;
        }
        if (token === "-m" || token === "--model") {
          flags.model = value();
          modelExplicit = true;
        } else if (token === "-p" || token === "--prompt") {
          flags.prompt = value();
        } else if (token === "-T" || token === "--tui") {
          flags.tui = true;
        } else if (token === "--system-prompt") {
          flags.systemPrompt = value();
          systemPromptExplicit = true;
        } else if (token === "--system-prompt-file") {
          systemPromptFile = value();
          systemPromptExplicit = true;
        } else if (token === "--reasoning") {
          const level = value() as ReasoningLevel;
          if (!REASONING_LEVELS.includes(level)) {
            process.stderr.write(
              `Invalid --reasoning: ${level} (use ${REASONING_LEVELS.join(", ")})\n`,
            );
            return 1;
          }
          flags.reasoning = level;
        } else if (token === "--mode") {
          const mode = value() as Mode;
          if (!MODES.includes(mode)) {
            process.stderr.write(`Invalid --mode: ${mode} (use ${MODES.join(", ")})\n`);
            return 1;
          }
          flags.mode = mode;
        } else if (token === "--no-web-search") {
          flags.webSearch = false;
        } else if (token === "--tools") {
          toolsExplicit = true;
          const raw = value();
          // "none" disables tools entirely (the way to opt out of the all-tools
          // default); otherwise the value is a comma-separated tool list.
          if (raw.trim() === "none") {
            flags.tools = [];
          } else {
            for (const name of raw.split(",")) {
              const trimmed = name.trim() as ToolName;
              if (trimmed.length === 0) {
                continue;
              }
              if (!TOOL_NAMES.includes(trimmed)) {
                process.stderr.write(`Unknown tool: ${trimmed} (use ${TOOL_NAMES.join(", ")})\n`);
                return 1;
              }
              if (!flags.tools.includes(trimmed)) {
                flags.tools.push(trimmed);
              }
            }
          }
        } else if (token === "--load") {
          flags.loadPath = value();
        } else if (token === "--save") {
          flags.savePath = value();
        } else {
          process.stderr.write(
            `Unknown argument: ${token}\nRun "${fullName(this, prefix)} -h" for usage.\n`,
          );
          return 1;
        }
      } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
        return 1;
      }
    }

    // Default to all tools when --tools was not given at all.
    if (!toolsExplicit) {
      flags.tools = [...TOOL_NAMES];
    }

    // Without -m, use the pinned default model from preferences (built-in
    // default if none pinned or the store is unavailable).
    if (!modelExplicit) {
      flags.model = loadPreferences().defaultModel ?? DEFAULT_DEFAULT_MODEL;
    }

    if (flags.prompt !== undefined && flags.tui) {
      process.stderr.write("--tui is ignored with --prompt (one-shot, non-interactive)\n");
    }

    registerOpenAINative();
    registerOpenRouterNative();
    // Shared with request_human_edit (via loadTools) so the editor can take the
    // terminal; the TUI replaces .suspend with a tui.stop/start wrapper.
    const terminal: TerminalController = { suspend: (fn) => fn() };
    let agent: Agent;
    try {
      let model: Model<Api>;
      try {
        model = resolveModel(flags.model);
      } catch (error) {
        // An explicit -m (or the built-in default) failing is a hard error; a
        // remembered model that no longer resolves falls back to the default.
        if (modelExplicit || flags.model === DEFAULT_DEFAULT_MODEL) {
          throw error;
        }
        process.stderr.write(
          `${dim(`pinned default model ${flags.model} unavailable (${error instanceof Error ? error.message : error}); using ${DEFAULT_DEFAULT_MODEL}`)}\n`,
        );
        flags.model = DEFAULT_DEFAULT_MODEL;
        model = resolveModel(flags.model);
      }
      if (systemPromptFile) {
        flags.systemPrompt = readFileSync(systemPromptFile, "utf8").trim();
      }
      const loaded = flags.loadPath ? loadContext(flags.loadPath) : undefined;
      // A loaded context supplies the system prompt unless a flag overrides it.
      const systemPrompt = systemPromptExplicit
        ? flags.systemPrompt
        : (loaded?.systemPrompt ?? flags.systemPrompt);
      agent = new Agent({
        initialState: {
          systemPrompt,
          model,
          thinkingLevel: flags.reasoning,
          messages: loaded?.messages ?? [],
          tools:
            flags.tools.length > 0
              ? await loadTools(flags.tools, process.cwd(), (fn) => terminal.suspend(fn))
              : [],
        },
        streamFn: (m, ctx, options) =>
          streamSimple(m, ctx, {
            ...options,
            webSearch: flags.webSearch,
          } as SimpleStreamOptions),
        convertToLlm: (messages) => {
          // Failed turns are dropped below; their tool results must be dropped
          // too, or a function_call_output is left orphaned and the API
          // rejects every subsequent request.
          const droppedToolCallIds = new Set<string>();
          for (const m of messages) {
            if (
              m.role === "assistant" &&
              (m.stopReason === "error" || m.stopReason === "aborted")
            ) {
              for (const b of m.content) {
                if (b.type === "toolCall") {
                  droppedToolCallIds.add(b.id);
                }
              }
            }
          }
          return messages.flatMap((m): Message[] => {
            if (m.role === "toolResult") {
              return droppedToolCallIds.has(m.toolCallId) ? [] : [m];
            }
            if (m.role === "user") {
              return [m];
            }
            if (m.role !== "assistant") {
              return [];
            }
            // Drop failed turns (empty content, unreplayable state) so a
            // transient error does not poison every later request.
            if (m.stopReason === "error" || m.stopReason === "aborted") {
              return [];
            }
            // Replay is lossless only to the EXACT producing model: the
            // provider converters key replayability on api+provider+model
            // (encrypted reasoning and hosted raw items are model-specific),
            // so two models sharing an api (e.g. openai-native/gpt-5-mini and
            // openai-native/gpt-5, or two openrouter-native models) are NOT
            // interchangeable. Match that triple here, or a cross-model switch
            // would skip flatten and the converter would then silently drop the
            // hosted history anyway. flattenHostedContent is provider-agnostic,
            // so it handles openrouter-native's hosted blocks the same way.
            const sameModel =
              m.api === agent.state.model.api &&
              m.provider === agent.state.model.provider &&
              m.model === agent.state.model.id;
            if (sameModel) {
              // The rendered citation trailer is a display/persistence artifact;
              // the producing provider reconstructs the real annotations itself,
              // so strip the trailer before replay. (For openai-native the
              // provider's converter also strips it defensively; doing it here
              // keeps the wire payload identical for same-model replay.)
              return [stripRenderedCitations(m)];
            }
            // Handoff to a different model (foreign provider, or another model
            // of the same provider): the target converter only understands
            // text | thinking | toolCall and would silently drop hostedToolCall
            // blocks, ignore inline citations, and reject the producing model's
            // thinking signatures. Flatten the extended content to plain text
            // the target preserves.
            return [flattenHostedContent(m)];
          });
        },
      });
    } catch (error) {
      process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
      return 1;
    }

    const save = () => {
      if (flags.savePath) {
        saveContext(agent, flags.savePath);
        process.stderr.write(`${dim(`context saved to ${flags.savePath}`)}\n`);
      }
    };

    if (flags.prompt !== undefined) {
      // One-shot streams the answer to stdout (scriptable; never paged).
      const renderer = attachRenderer(agent, flags, false);
      // One-shot is non-interactive: there is no human to confirm tool calls,
      // so slow mode denies them rather than running them unattended. go mode
      // (the default) keeps the pass-through, so -p behavior is unchanged.
      agent.beforeToolCall = makeBeforeToolCall(flags, async (toolName, argsPreview) => {
        process.stderr.write(
          `${dim(`run ${toolName} ${argsPreview}? [y/N] (non-interactive — denied)`)}\n`,
        );
        return false;
      });
      await agent.prompt(flags.prompt);
      save();
      return renderer.hadError() ? 1 : 0;
    }

    if (flags.tui) {
      const code = await runTui(agent, flags, terminal);
      save();
      return code;
    }

    // Interactive REPL: in slow mode, page long replies (pageReplies = true).
    const renderer = attachRenderer(agent, flags, true);
    process.stderr.write(
      `${dim(`model: ${flags.model} | mode: ${flags.mode} | /config for commands | Ctrl+C interrupts, Ctrl+D quits`)}\n`,
    );
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    // Slow-mode tool confirmation. The hook fires while agent.prompt() is being
    // awaited inside the loop below, so the `for await (line of rl)` iterator is
    // suspended and not competing for stdin; rl.question reads the next line
    // from the same interface, avoiding a second stdin reader (which would
    // corrupt line buffering). Non-TTY stdin (piped/-p) cannot answer a
    // prompt, so default to deny there — slow mode without a human present must
    // not auto-run tools.
    agent.beforeToolCall = makeBeforeToolCall(flags, async (toolName, argsPreview, signal) => {
      const question = `run ${toolName} ${argsPreview}? [y/N] `;
      if (!process.stdin.isTTY) {
        process.stderr.write(`${dim(`${question}(no TTY — denied)`)}\n`);
        return false;
      }
      try {
        // Ctrl+C during the prompt aborts the turn; the signal cancels the
        // question so we do not block on a line that will never come.
        const answer = (await rl.question(dim(question), { signal })).trim().toLowerCase();
        return answer === "y" || answer === "yes";
      } catch {
        return false; // aborted question -> deny.
      }
    });
    // Iterating the interface (rather than looping rl.question) queues input
    // arriving while a turn runs instead of dropping it. Ctrl+C aborts the
    // active response; a second Ctrl+C before it settles (or one while idle)
    // quits. Note: abort stops the in-flight model request, but an
    // already-running tool finishes, and one queued follow-up turn may still
    // start — the agent loop only observes the abort at request boundaries.
    let abortRequested = false;
    rl.on("SIGINT", () => {
      if (agent.state.isStreaming && !abortRequested) {
        abortRequested = true;
        agent.abort();
      } else {
        rl.close();
      }
    });
    agent.subscribe((event) => {
      if (event.type === "agent_start") {
        abortRequested = false;
      }
    });
    process.stderr.write("rath> ");
    try {
      for await (const line of rl) {
        const trimmed = line.trim();
        if (trimmed.length === 0) {
          process.stderr.write("rath> ");
          continue;
        }
        const result = await handleSlashCommand(trimmed, agent, flags, terminal);
        if (result) {
          if (result.output !== undefined) {
            const text = result.output.length > 0 ? result.output : "(empty)";
            // Slow mode pages long, non-error output (e.g. /sys, /lsmodels)
            // through $PAGER; pagePlain returns false (no-op) in go mode, for
            // short output, or with no usable pager, so we print as before.
            if (result.isError || !pagePlain(text, flags)) {
              process.stderr.write(`${result.isError ? text : dim(text)}\n`);
            }
          }
          if (result.exit) {
            break;
          }
          process.stderr.write("rath> ");
          continue;
        }
        await agent.prompt(trimmed);
        process.stderr.write("rath> ");
      }
    } finally {
      rl.close();
      save();
    }
    return renderer.hadError() ? 1 : 0;
  },
};

// ---------------------------------------------------------------------------
// TUI frontend (-T/--tui)
// ---------------------------------------------------------------------------

// Styles do not carry across TUI lines; every style helper applies per line.
function styled(open: string): (text: string) => string {
  return (text: string) =>
    text
      .split("\n")
      .map((line) => `${open}${line}\x1b[0m`)
      .join("\n");
}

const dimLine = styled("\x1b[2m");
const cyanLine = styled("\x1b[36m");
const yellowLine = styled("\x1b[33m");
const redLine = styled("\x1b[31m");

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function messageText(content: string | { type: string; text?: string }[]): string {
  if (typeof content === "string") {
    return content;
  }
  return content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("\n");
}

/**
 * pi-tui frontend. Same session semantics as the plain REPL — same Agent,
 * same slash commands, same citation merging — rendered with Pi's own
 * interactive components (AssistantMessageComponent, UserMessageComponent,
 * ToolExecutionComponent, and the Pi theme), so the TUI looks like vanilla
 * Pi. rath adds what those components do not know about: hosted-tool
 * markers and clickable citation sources. Messages stream as they arrive
 * and are re-rendered in full on completion.
 */
async function runTui(
  agent: Agent,
  flags: RunFlags,
  terminal: TerminalController,
): Promise<number> {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    process.stderr.write("--tui requires an interactive terminal\n");
    return 1;
  }

  // The TUI's dependencies load here so plain runs never pay for them.
  const piTui = await import("@earendil-works/pi-tui");
  const piCa = await import("@earendil-works/pi-coding-agent");
  const {
    Container,
    Editor,
    hyperlink,
    Loader,
    matchesKey,
    ProcessTerminal,
    SelectList,
    Text,
    truncateToWidth,
    TUI,
  } = piTui;
  const {
    AssistantMessageComponent,
    getMarkdownTheme,
    getSelectListTheme,
    initTheme,
    ToolExecutionComponent,
    UserMessageComponent,
  } = piCa;

  // Pi's built-in dark theme, selected explicitly. Caveat: initTheme also
  // registers (not applies) themes found in ~/.pi/agent/themes — the only
  // implicit read in rath run, cosmetic-only, forced on us because
  // pi-coding-agent does not export setThemeInstance/loadThemeFromPath.
  initTheme("dark", false);
  const markdownTheme = getMarkdownTheme();
  const selectTheme = getSelectListTheme();
  const editorTheme: PiTui.EditorTheme = { borderColor: dimLine, selectList: selectTheme };

  /** Clickable sources list rebuilt from the message's citations. */
  const sourcesComponent = (message: AssistantMessage): PiTui.Text | undefined => {
    const citations = uniqueUrlCitations(message);
    if (citations.length === 0) {
      return undefined;
    }
    const lines = citations.map(
      (c) => `  ${hyperlink(c.title || c.url, c.url)}${c.title ? dimLine(` — ${c.url}`) : ""}`,
    );
    return new Text(`${dimLine("sources:")}\n${lines.join("\n")}`);
  };

  const hostedMarkers = (message: AssistantMessage): PiTui.Text[] =>
    contentBlocks(message)
      .filter(isHostedToolCall)
      .map((block) => new Text(yellowLine(`[${block.toolName}]`)));

  const tui = new TUI(new ProcessTerminal());
  const transcript = new Container();
  const status = new Container();
  const loader = new Loader(tui, cyanLine, dimLine, "thinking…", {
    frames: SPINNER,
    intervalMs: 80,
  });
  const editor = new Editor(tui, editorTheme);

  const banner = new Text("");
  const refreshBanner = () => {
    banner.setText(
      `${cyanLine("rath")} ${dimLine(
        `| ${flags.model} | mode: ${flags.mode} | reasoning: ${agent.state.thinkingLevel} | ` +
          "/config for commands | Ctrl+C interrupts",
      )}`,
    );
  };
  refreshBanner();
  tui.addChild(banner);
  tui.addChild(transcript);
  tui.addChild(status);
  tui.addChild(editor);
  tui.setFocus(editor);

  const addAssistant = (message: AssistantMessage) => {
    transcript.addChild(
      new AssistantMessageComponent(stripRenderedCitations(message), false, markdownTheme),
    );
    for (const marker of hostedMarkers(message)) {
      transcript.addChild(marker);
    }
    const sources = sourcesComponent(message);
    if (sources) {
      transcript.addChild(sources);
    }
  };

  /** Overlay selector; Enter applies, Escape cancels. */
  const openSelector = (items: PiTui.SelectItem[], onPick: (value: string) => void) => {
    const list = new SelectList(items, 12, selectTheme);
    const handle = tui.showOverlay(list, { width: "80%", maxHeight: "60%" });
    list.onSelect = (item) => {
      handle.hide();
      onPick(item.value);
      tui.requestRender();
    };
    list.onCancel = () => {
      handle.hide();
      tui.requestRender();
    };
  };

  // Scrollable pager overlay for slow mode. A focused Component that wraps the
  // text to the overlay width and shows one viewport at a time; arrows / j,k /
  // PageUp,PageDown / space / g,G scroll, q or Escape dismiss. Used so long
  // output (long /sys, /lsmodels, large tool results) can be read without
  // scrolling the whole transcript past it.
  // Page long text the way the user expects a pager to behave: suspend the TUI
  // and hand the whole terminal to the real $PAGER (less), which gives
  // full-screen rendering and the full set of less/vim keybindings (arrows,
  // PgUp/PgDn, j/k, g/G, /search, q) for free — far better than a half-screen
  // overlay that the transcript bleeds through. spawnSync blocks until the
  // pager exits, so calls serialize naturally (no queue needed); afterward the
  // TUI is restarted and re-renders the transcript. Returns true if it paged.
  const pageInTui = (text: string): boolean => {
    tui.stop();
    let paged = false;
    try {
      paged = runExternalPager(text);
    } finally {
      tui.start();
      tui.requestRender();
    }
    return paged;
  };

  // request_human_edit's editor needs sole ownership of the terminal; suspend
  // the TUI for its duration the same way the pager does, then redraw.
  terminal.suspend = <T>(fn: () => T): T => {
    tui.stop();
    try {
      return fn();
    } finally {
      tui.start();
      tui.requestRender();
    }
  };

  // In slow mode, page long text; otherwise (or if no usable pager) add it
  // inline. The `addInline` callback owns the non-paged rendering so callers
  // keep their own styling (dim vs red, padding, etc.).
  const pageOrShow = (text: string, addInline: () => void): void => {
    if (flags.mode === "slow" && isLongOutput(text)) {
      if (!pageInTui(text)) {
        addInline();
      }
    } else {
      addInline();
    }
  };

  const showCommandOutput = (result: { output?: string; isError?: boolean }) => {
    if (result.output !== undefined) {
      const text = result.output.length > 0 ? result.output : "(empty)";
      const addInline = () =>
        transcript.addChild(new Text(result.isError ? redLine(text) : dimLine(text)));
      // Never page errors (short and worth keeping in the transcript).
      if (result.isError) {
        addInline();
      } else {
        pageOrShow(text, addInline);
      }
    }
  };
  const echoCommandResult = async (input: string) => {
    showCommandOutput((await handleSlashCommand(input, agent, flags, terminal)) ?? {});
    refreshBanner();
    tui.requestRender();
  };

  // Slow-mode tool-call confirmation overlay. A focused Component showing the
  // tool and an args preview; y approves, n / Escape denies. Resolves the
  // returned promise with the decision, which makeBeforeToolCall turns into a
  // pass-through (approve) or a block (deny). The agent loop awaits this hook,
  // so the tool does not run until the user answers.
  //
  // Ctrl+C is intercepted by the TUI input listener (below), not by this
  // overlay, so it aborts the agent and fires the abort signal — we listen for
  // it here to dismiss the overlay and deny, otherwise this promise would hang
  // the agent loop forever waiting for a y/n that an interrupted user expects
  // to be unnecessary.
  const confirmTool = (
    toolName: string,
    argsPreview: string,
    signal?: AbortSignal,
  ): Promise<boolean> =>
    new Promise<boolean>((resolve) => {
      let settled = false;
      const decide = (approved: boolean, reason: "user" | "abort" = "user") => {
        if (settled) {
          return;
        }
        settled = true;
        signal?.removeEventListener("abort", onAbort);
        // handle.hide() restores focus to the overlay underneath (e.g. an open
        // pager) or the editor; forcing the editor here would strand it.
        handle.hide();
        transcript.addChild(
          new Text(
            approved
              ? dimLine(`[approved ${toolName}]`)
              : yellowLine(`[denied ${toolName}${reason === "abort" ? " (interrupted)" : ""}]`),
          ),
        );
        tui.requestRender();
        resolve(approved);
      };
      const onAbort = () => decide(false, "abort");
      if (signal?.aborted) {
        // Already aborted before the overlay even shows: deny without opening.
        transcript.addChild(new Text(yellowLine(`[denied ${toolName} (interrupted)]`)));
        tui.requestRender();
        resolve(false);
        return;
      }
      signal?.addEventListener("abort", onAbort, { once: true });
      const prompt: PiTui.Component = {
        render: (width: number): string[] => {
          const head = yellowLine(`Run tool ${toolName}?`);
          const args = dimLine(`  args: ${argsPreview}`);
          const help = dimLine("  [y] approve   [n]/Esc deny");
          return [head, args, "", help].map((l) => truncateToWidth(l, width));
        },
        handleInput: (data: string): void => {
          if (matchesKey(data, "y") || matchesKey(data, "shift+y")) {
            decide(true);
          } else if (
            matchesKey(data, "n") ||
            matchesKey(data, "shift+n") ||
            matchesKey(data, "escape")
          ) {
            decide(false);
          }
        },
        invalidate: () => {},
      };
      const handle = tui.showOverlay(prompt, { width: "80%", maxHeight: 8 });
      handle.focus();
      tui.requestRender();
    });

  agent.beforeToolCall = makeBeforeToolCall(flags, (toolName, argsPreview, signal) =>
    confirmTool(toolName, argsPreview, signal),
  );

  /** TUI-native rendering for selected slash commands. Returns true when handled. */
  const handleTuiCommand = (input: string): boolean => {
    const [command = "", ...rest] = input.split(/\s+/);
    const arg = rest.join(" ").trim();
    if (command === "/model" && arg.length === 0) {
      openSelector(
        listModels().map((spec) => ({ value: spec, label: spec })),
        (spec) => echoCommandResult(`/model ${spec}`),
      );
      return true;
    }
    if (command === "/reasoning" && arg.length === 0) {
      openSelector(
        REASONING_LEVELS.map((level) => ({
          value: level,
          label: level,
          description: level === agent.state.thinkingLevel ? "current" : "",
        })),
        (level) => echoCommandResult(`/reasoning ${level}`),
      );
      return true;
    }
    // Bare /config renders the panel; /config <subcommand> falls through to
    // handleSlashCommand (default-model, etc.).
    if (command === "/config" && arg.length === 0) {
      const pairs = sessionInfo(agent, flags);
      const width = Math.max(...pairs.map(([key]) => key.length));
      const panel = pairs
        .map(([key, value]) => `${dimLine(key.padEnd(width))}  ${value}`)
        .join("\n");
      transcript.addChild(new Text(panel, 2, 1));
      return true;
    }
    return false;
  };

  // Replay a loaded context into the transcript so the session is visible.
  for (const message of agent.state.messages) {
    if (message.role === "user") {
      transcript.addChild(new UserMessageComponent(messageText(message.content), markdownTheme));
    } else if (message.role === "assistant") {
      addAssistant(message as AssistantMessage);
    }
  }

  let resolveDone: (code: number) => void = () => {};
  const done = new Promise<number>((resolve) => {
    resolveDone = resolve;
  });

  let current: PiCodingAgent.AssistantMessageComponent | null = null;
  const toolComponents = new Map<string, PiCodingAgent.ToolExecutionComponent>();
  agent.subscribe((event) => {
    if (event.type === "agent_start") {
      status.addChild(loader);
      loader.start();
    } else if (event.type === "agent_end") {
      loader.stop();
      status.clear();
      toolComponents.clear();
      // A prompt submitted during the final steering-poll window stays queued.
      // Drain it once the run has fully settled (continue() throws while the
      // run is still active), instead of waiting for the next manual prompt.
      if (agent.hasQueuedMessages()) {
        setImmediate(() => {
          if (!agent.state.isStreaming && agent.hasQueuedMessages()) {
            agent.continue().catch((error) => {
              transcript.addChild(
                new Text(redLine(`error: ${error instanceof Error ? error.message : error}`)),
              );
              tui.requestRender();
            });
          }
        });
      }
    } else if (event.type === "message_start") {
      if (event.message.role === "user") {
        transcript.addChild(
          new UserMessageComponent(messageText(event.message.content), markdownTheme),
        );
      } else if (event.message.role === "assistant") {
        // In slow mode the reply is not streamed into the transcript; it is
        // paged at message_end. Streaming it live AND opening a pager over the
        // same content would show it twice — the loader is the only progress
        // indicator while it generates.
        if (flags.mode !== "slow") {
          current = new AssistantMessageComponent(undefined, false, markdownTheme);
          transcript.addChild(current);
        }
      }
    } else if (event.type === "message_update") {
      if (event.message.role === "assistant") {
        current?.updateContent(stripRenderedCitations(event.message as AssistantMessage));
      }
    } else if (event.type === "message_end") {
      if (event.message.role === "assistant") {
        const message = event.message as AssistantMessage;
        const failed = message.stopReason === "error" || message.stopReason === "aborted";
        if (!failed) {
          applyCitationTrailer(message);
        }
        if (current) {
          // Replace the streaming component with the full final rendering
          // (markers and sources included).
          transcript.removeChild(current);
          current = null;
        }
        // pi-agent-core resolves prompt() normally on provider errors and
        // emits them here, not as a rejection — render the error explicitly
        // instead of an empty assistant bubble.
        if (failed) {
          transcript.addChild(new Text(redLine(`error: ${message.errorMessage ?? "unknown"}`)));
        } else {
          addAssistant(message);
          // Slow mode: page the whole turn (thinking, answer, sources) in the
          // real pager for deliberate reading; it stays in the transcript too.
          const turnText = buildPagedTurnText(message);
          if (flags.mode === "slow" && isLongOutput(turnText)) {
            pageInTui(turnText);
          }
        }
      }
    } else if (event.type === "tool_execution_start") {
      const component = new ToolExecutionComponent(
        event.toolName,
        event.toolCallId,
        event.args,
        undefined,
        undefined,
        tui,
        process.cwd(),
      );
      component.setArgsComplete();
      component.markExecutionStarted();
      toolComponents.set(event.toolCallId, component);
      transcript.addChild(component);
    } else if (event.type === "tool_execution_update") {
      const component = toolComponents.get(event.toolCallId);
      const partial = event.partialResult as
        | { content?: { type: string }[]; details?: unknown }
        | undefined;
      if (component && partial?.content) {
        component.updateResult(
          { content: partial.content, details: partial.details, isError: false },
          true,
        );
      }
    } else if (event.type === "tool_execution_end") {
      const component = toolComponents.get(event.toolCallId);
      const result = event.result as {
        content?: { type: string; text?: string }[];
        details?: unknown;
      };
      component?.updateResult(
        { content: result?.content ?? [], details: result?.details, isError: event.isError },
        false,
      );
      // Slow mode: offer the full tool result in the pager when it is long, so
      // large outputs can be read without the inline component truncating them.
      if (flags.mode === "slow" && !event.isError) {
        const body = (result?.content ?? [])
          .filter((c) => c.type === "text" && typeof c.text === "string")
          .map((c) => c.text as string)
          .join("\n");
        if (body.length > 0 && isLongOutput(body)) {
          pageInTui(body);
        }
      }
    }
    tui.requestRender();
  });

  editor.onSubmit = (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return;
    }
    if (trimmed.startsWith("/")) {
      transcript.addChild(new Text(`\n${cyanLine("›")} ${trimmed}`));
      if (!handleTuiCommand(trimmed)) {
        handleSlashCommand(trimmed, agent, flags, terminal).then((result) => {
          if (result) {
            showCommandOutput(result);
            refreshBanner();
            if (result.exit) {
              tui.stop();
              resolveDone(0);
            }
          }
          tui.requestRender();
        });
      }
      tui.requestRender();
      return;
    }
    const message = { role: "user" as const, content: trimmed, timestamp: Date.now() };
    // Steer while a turn is streaming, or when a message is already queued for
    // the post-run drain — prompting directly in the latter case would jump
    // ahead of the queued one and break submission order.
    if (agent.state.isStreaming || agent.hasQueuedMessages()) {
      agent.steer(message);
      transcript.addChild(new Text(dimLine(`(queued) › ${trimmed}`)));
      tui.requestRender();
      return;
    }
    agent.prompt(message).catch((error) => {
      transcript.addChild(
        new Text(redLine(`error: ${error instanceof Error ? error.message : error}`)),
      );
      tui.requestRender();
    });
  };

  tui.addInputListener((data) => {
    if (matchesKey(data, "ctrl+c")) {
      if (agent.state.isStreaming) {
        // Abort the active response and discard any queued steers, so an
        // interrupt does not surface a forgotten follow-up afterward.
        agent.clearAllQueues();
        agent.abort();
      } else {
        tui.stop();
        resolveDone(0);
      }
      return { consume: true };
    }
    return undefined;
  });

  tui.start();
  const code = await done;
  agent.abort();
  await agent.waitForIdle();
  return code;
}
