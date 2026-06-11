/**
 * Integration test: the request_human_edit client-side tool. Unlike the other
 * integration tests this one calls no API and needs no key — the "human" is a
 * fake editor script that mutates the file and exits, so the whole human-edit
 * round-trip runs unattended. It checks:
 *  1. A temp file seeded with `content` is opened, the edit is read back, the
 *     returned details carry the temp path, changed=true, and a unified diff.
 *  2. A given relative path resolves against cwd and is returned the same way
 *     (consistent shape with the temp case).
 *  3. An editor that makes no change yields changed=false and an empty diff.
 *  4. resolveEditorCommand honors $VISUAL over $EDITOR, splits a command with
 *     args, appends --wait for GUI editors (and not when already present), and
 *     throws a clear error when no editor is available.
 *
 * Exits 0 on success, 1 on failure. Skipped on Windows (the fake editor is a
 * POSIX shell script).
 */
import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { createRequestHumanEditTool, resolveEditorCommand } from "../index.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

/** Write a POSIX editor script that runs `body` ("$1" is the file). */
function fakeEditor(dir: string, name: string, body: string): string {
  const path = join(dir, name);
  writeFileSync(path, `#!/bin/sh\n${body}\n`);
  chmodSync(path, 0o755);
  return path;
}

/** Run `fn` with env vars set, restoring them afterward. */
function withEnv(vars: Record<string, string | undefined>, fn: () => void): void {
  const saved: Record<string, string | undefined> = {};
  for (const [k, v] of Object.entries(vars)) {
    saved[k] = process.env[k];
    if (v === undefined) {
      delete process.env[k];
    } else {
      process.env[k] = v;
    }
  }
  try {
    fn();
  } finally {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) {
        delete process.env[k];
      } else {
        process.env[k] = v;
      }
    }
  }
}

async function main(): Promise<void> {
  if (process.platform === "win32") {
    log("Skipped on Windows (fake editor is a POSIX shell script).");
    return;
  }

  const work = mkdtempSync(join(tmpdir(), "rath-edit-it-"));
  const appendEditor = fakeEditor(work, "append.sh", 'printf "\\nhuman line\\n" >> "$1"');
  const noopEditor = fakeEditor(work, "noop.sh", ":");

  // Case 1: temp file + seeded content.
  {
    const tool = createRequestHumanEditTool({ cwd: work, editor: appendEditor });
    const result = await tool.execute("c1", { content: "draft one\ndraft two\n" });
    const { path, changed, diff, temp } = result.details;
    assert.equal(temp, true, "no path given -> temp file");
    assert.equal(changed, true, "append editor changes the file");
    assert.ok(diff.includes("+human line"), "diff shows the added line");
    assert.equal(readFileSync(path, "utf8"), "draft one\ndraft two\n\nhuman line\n");
    const text = result.content[0];
    assert.ok(text?.type === "text" && text.text.includes("human line"), "model sees final text");
    assert.ok(text.type === "text" && text.text.includes("Diff:"), "model sees the diff");
  }
  log("Case 1 OK: temp file seeded, edited, diff + final content returned");

  // Case 2: given relative path, no seed (created empty, then edited).
  {
    const tool = createRequestHumanEditTool({ cwd: work, editor: appendEditor });
    const result = await tool.execute("c2", { path: "notes.md" });
    assert.equal(result.details.temp, false, "given path -> not temp");
    assert.equal(result.details.path, join(work, "notes.md"), "relative path resolved against cwd");
    assert.equal(result.details.changed, true);
    assert.equal(readFileSync(join(work, "notes.md"), "utf8"), "\nhuman line\n");
  }
  log("Case 2 OK: given path resolved and returned, same shape as temp case");

  // Case 3: editor makes no change.
  {
    const tool = createRequestHumanEditTool({ cwd: work, editor: noopEditor });
    const result = await tool.execute("c3", { content: "unchanged\n" });
    assert.equal(result.details.changed, false, "no-op editor -> changed=false");
    assert.equal(result.details.diff, "", "no change -> empty diff");
  }
  log("Case 3 OK: no-op editor reports changed=false, empty diff");

  // Case 4: editor resolution rules.
  withEnv({ VISUAL: "vim", EDITOR: "code" }, () => {
    assert.deepEqual(resolveEditorCommand(), ["vim"], "VISUAL wins over EDITOR");
  });
  withEnv({ VISUAL: undefined, EDITOR: "code" }, () => {
    assert.deepEqual(resolveEditorCommand(), ["code", "--wait"], "GUI editor gets --wait");
  });
  withEnv({ VISUAL: undefined, EDITOR: "code --wait" }, () => {
    assert.deepEqual(resolveEditorCommand(), ["code", "--wait"], "--wait not duplicated");
  });
  withEnv({ VISUAL: undefined, EDITOR: "cursor" }, () => {
    assert.deepEqual(resolveEditorCommand(), ["cursor", "--wait"], "cursor gets --wait");
  });
  withEnv({ VISUAL: undefined, EDITOR: "nano" }, () => {
    assert.deepEqual(resolveEditorCommand(), ["nano"], "terminal editor unchanged");
  });
  // No editor anywhere: empty PATH (plus no VISUAL/EDITOR) must throw.
  {
    const emptyDir = join(work, "empty-path");
    mkdirSync(emptyDir, { recursive: true });
    withEnv({ VISUAL: undefined, EDITOR: undefined, PATH: emptyDir + delimiter }, () => {
      assert.throws(() => resolveEditorCommand(), /No editor found/, "throws when none available");
    });
  }
  log("Case 4 OK: VISUAL/EDITOR precedence, --wait injection, and no-editor error");

  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
