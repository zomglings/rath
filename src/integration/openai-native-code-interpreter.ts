/**
 * Integration test for the hosted code interpreter tool on the openai-native
 * provider. Verifies that:
 *  1. An opt-in hosted tool (codeInterpreter) is sent and its raw
 *     `code_interpreter_call` output item is captured as a hostedToolCall
 *     block.
 *  2. The raw item is replayed without loss on a follow-up turn, after the
 *     context has been JSON serialize/deserialized.
 *
 * Requires OPENAI_API_KEY. Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import { type AssistantMessage, type Context, stream } from "@earendil-works/pi-ai";
import {
  getHostedToolCalls,
  type HostedToolCallItem,
  type OpenAINativeOptions,
  openaiNativeModel,
  registerOpenAINative,
} from "../index.js";

const MODEL_ID = process.env.RATH_TEST_MODEL || "gpt-5-mini";
const FIB_100 = "354224848179261915075";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function runTurn(context: Context, options?: OpenAINativeOptions): Promise<AssistantMessage> {
  const model = openaiNativeModel(MODEL_ID);
  const events = stream(model, context, {
    reasoningEffort: "low",
    webSearch: false,
    codeInterpreter: true,
    ...options,
  });
  const message = await events.result();
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    throw new Error(`Turn failed (${message.stopReason}): ${message.errorMessage}`);
  }
  return message;
}

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  registerOpenAINative();

  const context: Context = {
    systemPrompt: "You are a precise computational assistant. Use the code interpreter for math.",
    messages: [],
  };

  // ---- Turn 1: trigger the hosted code interpreter ----
  log(`Turn 1 (model: ${MODEL_ID}): computing fib(100) with the code interpreter`);
  context.messages.push({
    role: "user",
    content:
      "Use the code interpreter to compute the 100th Fibonacci number exactly " +
      "(fib(1) = fib(2) = 1). Reply with just the number.",
    timestamp: Date.now(),
  });
  const turn1 = await runTurn(context);

  const calls1 = getHostedToolCalls<HostedToolCallItem>(turn1);
  assert.ok(calls1.length > 0, "turn 1 must contain at least one hostedToolCall block");
  for (const call of calls1) {
    assert.equal(call.toolName, "code_interpreter");
    assert.equal(call.raw.type, "code_interpreter_call");
    assert.equal(call.status, "completed", "code interpreter call must complete");
  }
  const text1 = turn1.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
  assert.ok(text1.includes(FIB_100), `reply must contain fib(100) (got: ${text1})`);
  log(`Turn 1 OK: ${calls1.length} code_interpreter_call(s), correct result`);
  context.messages.push(turn1);

  // ---- Turn 2: JSON round-trip, then replay the raw items without loss ----
  log("Turn 2: JSON round-trip, then verifying raw item replay");
  const restored = JSON.parse(JSON.stringify(context)) as Context;
  assert.deepEqual(restored, context, "context must survive JSON serialize/deserialize");

  restored.messages.push({
    role: "user",
    content: "Without running any more code: what is the last digit of that number?",
    timestamp: Date.now(),
  });
  let replayedItems: unknown[] = [];
  const turn2 = await runTurn(restored, {
    onPayload: (payload) => {
      const params = payload as { input: { type?: string }[] };
      replayedItems = params.input.filter((item) => item.type === "code_interpreter_call");
      return undefined;
    },
  });

  assert.deepEqual(
    replayedItems,
    calls1.map((c) => c.raw),
    "replayed code_interpreter_call items must match the captured raw items exactly",
  );
  const text2 = turn2.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
  assert.ok(text2.includes("5"), `reply must contain the last digit 5 (got: ${text2})`);
  log(`Turn 2 OK: ${replayedItems.length} raw item(s) replayed without loss`);

  const totalCost = [turn1, turn2].reduce((sum, m) => sum + m.usage.cost.total, 0);
  log(`All assertions passed. Total token cost: $${totalCost.toFixed(4)} (excludes container fee)`);
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
