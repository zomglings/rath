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
import { registerOpenAINative, registerOpenRouterNative } from "../index.js";

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

  // Case 1+2: apply every field, including a tools rebuild that keeps configure.
  const r1 = await configure.execute("c1", {
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
  assert.equal(r1.details.changes.length, 6, "all six changes recorded");
  assert.equal(r1.details.errors.length, 0, "no errors");
  log("Case 1+2 OK: all fields applied; tool set rebuilt including configure");

  // Case 3: invalid model and unknown tool are reported, valid field still applies.
  const r3 = await configure.execute("c3", {
    model: "not-a-spec",
    tools: ["read", "bogus"],
    reasoning: "medium",
  });
  assert.equal(r3.details.errors.length, 2, "model + tools errors reported");
  assert.ok(
    r3.details.errors.some((e: string) => e.startsWith("model")),
    "model error present",
  );
  assert.ok(
    r3.details.errors.some((e: string) => e.startsWith("tools")),
    "tools error present",
  );
  assert.equal(flags.reasoning, "medium", "valid field still applied alongside errors");
  assert.equal(flags.model, "openrouter-native/openai/gpt-4o", "bad model did not change flags");
  assert.deepEqual(flags.tools, ["read", "configure"], "bad tools list did not change tools");
  log("Case 3 OK: bad values reported per-field; valid fields still apply");

  // Case 4: empty call is a no-op but still returns config.
  const r4 = await configure.execute("c4", {});
  assert.equal(r4.details.changes.length, 0, "no changes");
  const text = r4.content[0];
  assert.ok(
    text?.type === "text" && text.text.includes("Current configuration:"),
    "returns current configuration",
  );
  log("Case 4 OK: empty call is a no-op that still reports configuration");

  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
