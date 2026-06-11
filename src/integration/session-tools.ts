/**
 * Integration test: the session-operating tools list_models, save_context, and
 * end_session. Calls no API and needs no key — it builds the tools via
 * loadTools with a stub agent / flags / requestExit and invokes execute()
 * directly. Checks:
 *  1. list_models returns the model catalog and honors a substring filter.
 *  2. save_context writes the session JSON to the given path and records it as
 *     the save-on-exit path.
 *  3. end_session calls requestExit and asks the loop to terminate.
 *
 * Exits 0 on success, 1 on failure. $RATH_CONFIG_DIR is pointed at a temp dir.
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Agent } from "@earendil-works/pi-agent-core";
import { loadTools, type RunFlags, resolveModel } from "../commands/run.js";
import { registerOpenAINative, registerOpenRouterNative } from "../index.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function main(): Promise<void> {
  const work = mkdtempSync(join(tmpdir(), "rath-session-it-"));
  process.env.RATH_CONFIG_DIR = work;
  registerOpenAINative();
  registerOpenRouterNative();

  const state = {
    model: resolveModel("openai-native/gpt-5.5"),
    thinkingLevel: "low",
    tools: [] as unknown[],
    systemPrompt: "be terse",
    messages: [{ role: "user", content: "hello", timestamp: 0 }] as unknown[],
  };
  const agent = { state } as unknown as Agent;
  const flags: RunFlags = {
    model: "openai-native/gpt-5.5",
    systemPrompt: "be terse",
    reasoning: "low",
    webSearch: true,
    tools: ["list_models", "save_context", "end_session"],
    mode: "go",
  };
  let exitCalls = 0;
  const tools = await loadTools(flags.tools, process.cwd(), {
    getAgent: () => agent,
    flags,
    requestExit: () => {
      exitCalls++;
    },
  });
  const byName = new Map(tools.map((t) => [t.name, t]));
  assert.ok(
    byName.has("list_models") && byName.has("save_context") && byName.has("end_session"),
    "all three session tools were built",
  );

  // Case 1: list_models, unfiltered then filtered.
  const all = await byName.get("list_models")!.execute("list-all", {});
  assert.ok(all.details.models.length > 0, "lists some models");
  const filtered = await byName.get("list_models")!.execute("list-filtered", { filter: "gpt-5.5" });
  assert.ok(filtered.details.models.length > 0, "filter matches");
  assert.ok(
    filtered.details.models.every((m: string) => m.includes("gpt-5.5")),
    "every filtered model matches the substring",
  );
  assert.ok(filtered.details.models.length < all.details.models.length, "filter narrows the list");
  log("Case 1 OK: list_models enumerates and filters the catalog");

  // Case 2: save_context writes JSON and sets the save-on-exit path.
  const outPath = join(work, "session.json");
  const saved = await byName.get("save_context")!.execute("save", { path: outPath });
  assert.equal(saved.details.path, outPath);
  assert.equal(flags.savePath, outPath, "save path recorded for save-on-exit");
  const parsed = JSON.parse(readFileSync(outPath, "utf8"));
  assert.equal(parsed.systemPrompt, "be terse", "system prompt persisted");
  assert.equal(parsed.messages.length, 1, "messages persisted");
  log("Case 2 OK: save_context writes session JSON and sets save-on-exit");

  // Case 3: end_session requests exit and terminates the loop.
  const ended = await byName.get("end_session")!.execute("end", {});
  assert.equal(exitCalls, 1, "requestExit called once");
  assert.equal(ended.terminate, true, "tool result asks the agent loop to stop");
  log("Case 3 OK: end_session requests exit and terminates");

  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
