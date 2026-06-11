/**
 * Integration test: slow-mode tool gating freezes the mode per turn. No API.
 *
 * The security property: in slow mode every tool call is confirmed. A tool that
 * flips mode to "go" mid-turn (configure({mode:"go"})) must NOT disable the gate
 * for the rest of that same turn — otherwise the model self-approves one mode
 * change and then runs everything else unconfirmed. The gate snapshots the mode
 * at agent_start; a mid-turn flags.mode change only takes effect next turn.
 *
 * Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import type { Agent } from "@earendil-works/pi-agent-core";
import { makeBeforeToolCall, type RunFlags } from "../commands/run.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function main(): Promise<void> {
  // Stub agent that just records agent lifecycle subscribers.
  const handlers: Array<(event: { type: string }) => void> = [];
  const agent = {
    subscribe: (fn: (event: { type: string }) => void) => {
      handlers.push(fn);
    },
  } as unknown as Agent;
  const emit = (type: string) => {
    for (const h of handlers) {
      h({ type });
    }
  };

  const flags: RunFlags = {
    model: "openai-native/gpt-5.5",
    systemPrompt: "x",
    reasoning: "low",
    webSearch: true,
    tools: [],
    mode: "slow",
  };
  // Confirm always denies, so a gated call returns a block and a pass-through
  // (go mode) returns undefined — easy to distinguish.
  const hook = makeBeforeToolCall(agent, flags, async () => false);
  const callBash = () => hook({ toolCall: { name: "bash" }, args: {} } as never);

  // Turn 1 starts in slow mode.
  emit("agent_start");
  // Model flips mode to go mid-turn (as configure would).
  flags.mode = "go";
  const midTurn = await callBash();
  assert.ok(
    midTurn && (midTurn as { block?: boolean }).block === true,
    "mid-turn mode flip must NOT disarm the gate; the call is still blocked",
  );
  log("Case 1 OK: configure({mode:'go'}) mid-turn cannot disable gating for that turn");

  // Next turn picks up the new mode.
  emit("agent_start");
  const nextTurn = await callBash();
  assert.equal(nextTurn, undefined, "after the next agent_start, go mode passes through");
  log("Case 2 OK: the mode change takes effect on the following turn");

  // And flipping back to slow between turns re-gates.
  flags.mode = "slow";
  emit("agent_start");
  const reGated = await callBash();
  assert.ok(
    reGated && (reGated as { block?: boolean }).block === true,
    "slow between turns re-enables gating",
  );
  log("Case 3 OK: /slow between turns re-enables gating");

  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
