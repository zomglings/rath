/**
 * Integration test: the Barbarian Reviewer's usage/cost aggregation. No API,
 * no key.
 *
 * Covers totalUsage, the transcript -> aggregate reduction behind
 * BarbarianResult.usage and the CLI cost line: non-assistant messages are
 * ignored, every assistant turn (including errored ones) is summed, the
 * optional cacheWrite1h split only appears when a turn reported it, and a
 * checkpoint-restored message without usage is skipped rather than crashing.
 *
 * Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage } from "@earendil-works/pi-ai/compat";
import { totalUsage } from "../agents/barbarian.js";
import * as publicApi from "../index.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

function assistant(usage: Usage, stopReason: AssistantMessage["stopReason"]): AssistantMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text: "x" }],
    api: "openai-completions",
    provider: "openrouter",
    model: "openai/gpt-5.5",
    usage,
    stopReason,
    timestamp: 0,
  };
}

function usage(partial: Partial<Usage> & { cost: Usage["cost"] }): Usage {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    ...partial,
  };
}

async function main(): Promise<void> {
  // Case 1: an empty transcript sums to all-zero usage with no cacheWrite1h.
  const empty = totalUsage([]);
  assert.deepEqual(
    empty,
    {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    "empty transcript sums to zero",
  );
  assert.ok(!("cacheWrite1h" in empty), "cacheWrite1h absent when never reported");
  log("Case 1 OK: empty transcript");

  // Case 2: a mixed transcript — user and toolResult messages contribute
  // nothing; both assistant turns (one errored: spend before failure is real)
  // are summed across tokens and every cost component.
  const transcript: AgentMessage[] = [
    { role: "user", content: "review", timestamp: 0 },
    assistant(
      usage({
        input: 100,
        output: 10,
        cacheRead: 1000,
        cacheWrite: 50,
        totalTokens: 1160,
        cost: { input: 0.01, output: 0.02, cacheRead: 0.001, cacheWrite: 0.002, total: 0.033 },
      }),
      "toolUse",
    ),
    {
      role: "toolResult",
      toolCallId: "t1",
      toolName: "bash",
      content: [{ type: "text", text: "ok" }],
      isError: false,
      timestamp: 0,
    },
    assistant(
      usage({
        input: 200,
        output: 20,
        cacheRead: 2000,
        cacheWrite: 100,
        cacheWrite1h: 40,
        totalTokens: 2320,
        cost: { input: 0.02, output: 0.04, cacheRead: 0.002, cacheWrite: 0.004, total: 0.066 },
      }),
      "error",
    ),
  ];
  const total = totalUsage(transcript);
  assert.equal(total.input, 300, "input summed");
  assert.equal(total.output, 30, "output summed");
  assert.equal(total.cacheRead, 3000, "cacheRead summed");
  assert.equal(total.cacheWrite, 150, "cacheWrite summed");
  assert.equal(total.cacheWrite1h, 40, "cacheWrite1h carried from the turn that reported it");
  assert.equal(total.totalTokens, 3480, "totalTokens summed");
  assert.ok(Math.abs(total.cost.total - 0.099) < 1e-9, "cost.total summed (errored turn counts)");
  assert.ok(Math.abs(total.cost.input - 0.03) < 1e-9, "cost.input summed");
  assert.ok(Math.abs(total.cost.output - 0.06) < 1e-9, "cost.output summed");
  assert.ok(Math.abs(total.cost.cacheRead - 0.003) < 1e-9, "cost.cacheRead summed");
  assert.ok(Math.abs(total.cost.cacheWrite - 0.006) < 1e-9, "cost.cacheWrite summed");
  log("Case 2 OK: mixed transcript (non-assistant ignored, errored turn counted)");

  // Case 3: a checkpoint-restored assistant message with no usage field (old
  // format, hand-edited JSON) is skipped, not a crash.
  const legacy = { ...assistant(usage({ cost: empty.cost }), "stop") } as Record<string, unknown>;
  delete legacy.usage;
  const survived = totalUsage([legacy as unknown as AgentMessage]);
  assert.equal(survived.totalTokens, 0, "usage-less assistant message contributes nothing");
  log("Case 3 OK: usage-less assistant message skipped");

  // Case 4: negative cost components — pi-ai's bundled registry preserves
  // OpenRouter's -1/token sentinel for dynamic-priced auto-routers (e.g.
  // openrouter/auto), and calculateCost multiplies it into negative per-turn
  // costs. The aggregate clamps them to zero ("pricing unknown", not a
  // refund), and a mixed-sign turn's total is the sum of the clamped
  // components, never the raw negative total.
  const sentinel = totalUsage([
    assistant(
      usage({
        input: 100,
        output: 10,
        totalTokens: 110,
        cost: { input: -100, output: -10, cacheRead: 0, cacheWrite: 0, total: -110 },
      }),
      "stop",
    ),
    assistant(
      usage({
        input: 100,
        output: 10,
        totalTokens: 110,
        cost: { input: 0.05, output: -10, cacheRead: 0, cacheWrite: 0, total: -9.95 },
      }),
      "stop",
    ),
  ]);
  assert.equal(sentinel.cost.input, 0.05, "negative input cost clamped, priced turn kept");
  assert.equal(sentinel.cost.output, 0, "negative output cost clamped");
  assert.equal(sentinel.cost.total, 0.05, "total is the sum of clamped components");
  assert.equal(sentinel.totalTokens, 220, "token counts unaffected by cost clamping");
  log("Case 4 OK: negative pricing sentinels clamped to zero");

  // Case 5: the README-documented programmatic API is actually exported from
  // the package root (the only subpath package.json exposes).
  assert.equal(
    typeof publicApi.runBarbarianReview,
    "function",
    "runBarbarianReview exported from the package root",
  );
  assert.equal(publicApi.totalUsage, totalUsage, "totalUsage exported from the package root");
  assert.equal(
    typeof publicApi.hasCheckpoint,
    "function",
    "hasCheckpoint exported from the package root",
  );
  log("Case 5 OK: programmatic API exported from the package root");

  log("All assertions passed.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
