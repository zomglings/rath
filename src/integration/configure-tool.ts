/**
 * Integration test: the `configure` client-side tool. Calls no API and needs
 * no key — it builds the tool via loadTools (the real wiring) with a stub agent
 * and a flags object, then invokes execute() directly and asserts the session
 * mutations. Checks:
 *  1. Every field applies: model, reasoning, web search, mode, tools, system
 *     prompt — to both agent.state and flags, and the result lists the changes.
 *  2. configure can rebuild the tool set (including itself) via loadTools.
 *  3. Bad values are reported as per-field errors, not thrown, and valid
 *     fields in the same call still apply.
 *  4. An empty call is a no-op that still returns the current configuration.
 *
 * Exits 0 on success, 1 on failure. $RATH_CONFIG_DIR is pointed at a temp dir
 * so the test never touches the real preferences store.
 */
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Agent } from "@earendil-works/pi-agent-core";
import { loadTools, type RunFlags, resolveModel } from "../commands/run.js";
import { loadPreferences, registerOpenAINative, registerOpenRouterNative } from "../index.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function main(): Promise<void> {
  process.env.RATH_CONFIG_DIR = mkdtempSync(join(tmpdir(), "rath-configure-it-"));
  registerOpenAINative();
  registerOpenRouterNative();

  // Minimal stub agent: configure and sessionInfo only touch agent.state.
  const state = {
    model: resolveModel("openai-native/gpt-5.5"),
    thinkingLevel: "low",
    tools: [] as unknown[],
    systemPrompt: "original prompt",
    messages: [] as unknown[],
  };
  const agent = { state } as unknown as Agent;
  const flags: RunFlags = {
    model: "openai-native/gpt-5.5",
    systemPrompt: "original prompt",
    reasoning: "low",
    webSearch: true,
    tools: ["configure"],
    tui: false,
    mode: "go",
  };

  const tools = await loadTools(["configure"], process.cwd(), { getAgent: () => agent, flags });
  const configure = tools[0];
  assert.ok(configure && configure.name === "configure", "configure tool was built");

  // Apply every field at once, including a tools rebuild that keeps configure.
  const appliedAll = await configure.execute("apply-all", {
    model: "openrouter-native/openai/gpt-4o",
    reasoning: "high",
    webSearch: false,
    mode: "slow",
    tools: ["read", "configure"],
    systemPrompt: "you are a careful editor",
  });
  assert.equal(flags.model, "openrouter-native/openai/gpt-4o", "model applied to flags");
  assert.equal(flags.reasoning, "high");
  assert.equal(state.thinkingLevel, "high", "reasoning applied to agent.state");
  assert.equal(flags.webSearch, false);
  assert.equal(flags.mode, "slow");
  assert.equal(flags.systemPrompt, "you are a careful editor");
  assert.equal(state.systemPrompt, "you are a careful editor");
  assert.deepEqual(flags.tools, ["read", "configure"], "tool set replaced");
  assert.deepEqual(
    (state.tools as Array<{ name: string }>).map((t) => t.name),
    ["read", "configure"],
    "agent.state.tools rebuilt (read + configure)",
  );
  assert.equal(appliedAll.details.changes.length, 6, "all six changes recorded");
  assert.equal(appliedAll.details.errors.length, 0, "no errors");
  log("OK: all fields applied; tool set rebuilt including configure");

  // Invalid model and unknown tool are reported; the valid field still applies.
  const withBadValues = await configure.execute("bad-values", {
    model: "not-a-spec",
    tools: ["read", "bogus"],
    reasoning: "medium",
  });
  assert.equal(withBadValues.details.errors.length, 2, "model + tools errors reported");
  assert.ok(
    withBadValues.details.errors.some((e: string) => e.startsWith("model")),
    "model error present",
  );
  assert.ok(
    withBadValues.details.errors.some((e: string) => e.startsWith("tools")),
    "tools error present",
  );
  assert.equal(flags.reasoning, "medium", "valid field still applied alongside errors");
  assert.equal(flags.model, "openrouter-native/openai/gpt-4o", "bad model did not change flags");
  assert.deepEqual(flags.tools, ["read", "configure"], "bad tools list did not change tools");
  log("OK: bad values reported per-field; valid fields still apply");

  // An empty call is a no-op but still returns the configuration.
  const emptyRead = await configure.execute("empty", {});
  assert.equal(emptyRead.details.changes.length, 0, "no changes");
  const emptyReadText = emptyRead.content[0];
  assert.ok(
    emptyReadText?.type === "text" && emptyReadText.text.includes("Current configuration:"),
    "returns current configuration",
  );
  log("OK: empty call is a no-op that still reports configuration");

  // defaultModel pins/clears the persisted default (distinct from model).
  const sessionModelBefore = flags.model;
  await configure.execute("pin-default", { defaultModel: "openai-native/gpt-5.5" });
  assert.equal(loadPreferences().defaultModel, "openai-native/gpt-5.5", "default model pinned");
  assert.equal(flags.model, sessionModelBefore, "defaultModel does not change the session model");
  await configure.execute("clear-default", { defaultModel: "none" });
  assert.equal(loadPreferences().defaultModel, undefined, "default model cleared");
  const withBadDefault = await configure.execute("bad-default", { defaultModel: "not-a-spec" });
  assert.ok(
    withBadDefault.details.errors.some((e: string) => e.startsWith("defaultModel")),
    "invalid default model reported",
  );
  log("OK: defaultModel pins/clears the persisted default, session model untouched");

  // Out-of-set reasoning/mode are rejected (not written) and reported.
  const modeBefore = flags.mode;
  const reasoningBefore = flags.reasoning;
  const withInvalidEnums = await configure.execute("invalid-enums", {
    reasoning: "JUNK",
    mode: "JUNK",
  });
  assert.equal(withInvalidEnums.details.errors.length, 2, "both invalid enums reported");
  assert.equal(flags.mode, modeBefore, "invalid mode not applied");
  assert.equal(flags.reasoning, reasoningBefore, "invalid reasoning not applied");
  log("OK: invalid reasoning/mode rejected, not written, reported as errors");

  // A tools rebuild reports what was actually built. With requestExit in ctx,
  // end_session survives; without it, it is dropped and reported.
  let exitCalls = 0;
  const toolsWithExit = await loadTools(["configure"], process.cwd(), {
    getAgent: () => agent,
    flags,
    requestExit: () => {
      exitCalls++;
    },
  });
  const configureWithExit = toolsWithExit[0]!;
  const rebuiltWithExit = await configureWithExit.execute("rebuild-with-exit", {
    tools: ["configure", "end_session"],
  });
  assert.deepEqual(flags.tools, ["configure", "end_session"], "end_session kept when ctx has it");
  assert.equal(rebuiltWithExit.details.errors.length, 0, "no drop reported when end_session wired");
  assert.equal(exitCalls, 0, "building end_session does not call requestExit");

  // The original `configure` was built WITHOUT requestExit; its rebuild drops it.
  const rebuiltWithoutExit = await configure.execute("rebuild-without-exit", {
    tools: ["configure", "end_session"],
  });
  assert.deepEqual(flags.tools, ["configure"], "end_session dropped when ctx lacks requestExit");
  assert.ok(
    rebuiltWithoutExit.details.errors.some((e: string) => e.includes("end_session")),
    "the dropped tool is reported, not silently lost",
  );
  log("OK: tools rebuild reports the actually-built set; drops are surfaced");

  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
