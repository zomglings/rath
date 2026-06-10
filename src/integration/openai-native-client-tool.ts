/**
 * Integration test: client-side (function) tool calls through openai-native,
 * parsed by pi-ai's stream(). No agent loop involved — this checks the
 * provider's conversion layer only:
 *  1. The model's function call is parsed into a pi-ai ToolCall block
 *     (name, structured arguments, callId|itemId id) and the message stops
 *     with stopReason "toolUse".
 *  2. The tool executes client-side (prints hello world).
 *  3. On the next turn the function_call and function_call_output items are
 *     replayed to the API and the model acknowledges the result.
 *
 * Requires OPENAI_API_KEY. Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import { stream, Type, type AssistantMessage, type Context } from "@earendil-works/pi-ai";
import {
  getHostedToolCalls,
  openaiNativeModel,
  registerOpenAINative,
  type OpenAINativeOptions,
} from "../index.js";

const MODEL_ID = process.env.RATH_TEST_MODEL || "gpt-5-mini";

function log(message: string): void {
  process.stdout.write(message + "\n");
}

async function runTurn(context: Context, options?: OpenAINativeOptions): Promise<AssistantMessage> {
  const model = openaiNativeModel(MODEL_ID);
  const events = stream(model, context, { reasoningEffort: "low", webSearch: false, ...options });
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
    systemPrompt: "You are a terse assistant. Use the print tool when asked to print something.",
    messages: [],
    tools: [
      {
        name: "print",
        description: "Print a message to the user's terminal.",
        parameters: Type.Object({ message: Type.String({ description: "The text to print" }) }),
      },
    ],
  };

  // ---- Turn 1: the model calls the client-side tool; pi parses it ----
  log(`Turn 1 (model: ${MODEL_ID}): asking the model to call the print tool`);
  context.messages.push({
    role: "user",
    content: 'Print exactly "hello world" using the print tool.',
    timestamp: Date.now(),
  });
  let requestTools: { type?: string; name?: string }[] = [];
  const turn1 = await runTurn(context, {
    onPayload: (payload) => {
      const params = payload as { tools?: { type?: string; name?: string }[] };
      requestTools = params.tools ?? [];
      return undefined;
    },
  });

  assert.ok(
    requestTools.every((t) => t.type === "function"),
    `request must contain only function tools (got: ${requestTools.map((t) => t.type).join(", ")})`,
  );
  assert.equal(turn1.stopReason, "toolUse", "turn 1 must stop for tool use");
  assert.equal(getHostedToolCalls(turn1).length, 0, "no hostedToolCall blocks expected");

  const toolCalls = turn1.content.filter((b) => b.type === "toolCall");
  assert.equal(toolCalls.length, 1, "turn 1 must contain exactly one toolCall block");
  const toolCall = toolCalls[0]!;
  assert.equal(toolCall.name, "print");
  assert.ok(toolCall.id.includes("|"), `id must be callId|itemId (got: ${toolCall.id})`);
  assert.equal(typeof toolCall.arguments.message, "string", "arguments must be parsed JSON");
  assert.ok(
    toolCall.arguments.message.toLowerCase().includes("hello world"),
    `argument must be the requested text (got: ${toolCall.arguments.message})`,
  );
  assert.ok(
    !("partialJson" in toolCall),
    "the streaming scratch buffer must not survive on the final block",
  );
  log(`Turn 1 OK: toolCall parsed (name=print, message=${JSON.stringify(toolCall.arguments.message)})`);

  // ---- Execute the tool client-side ----
  log(`[print tool] ${toolCall.arguments.message}`);
  context.messages.push(turn1);
  context.messages.push({
    role: "toolResult",
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    content: [{ type: "text", text: "Printed successfully." }],
    isError: false,
    timestamp: Date.now(),
  });

  // ---- Turn 2: function_call + function_call_output replay ----
  log("Turn 2: replaying the tool call and result");
  let replayedCalls: { call_id?: string }[] = [];
  let replayedOutputs: { call_id?: string; output?: unknown }[] = [];
  const turn2 = await runTurn(context, {
    onPayload: (payload) => {
      const params = payload as { input: { type?: string; call_id?: string; output?: unknown }[] };
      replayedCalls = params.input.filter((item) => item.type === "function_call");
      replayedOutputs = params.input.filter((item) => item.type === "function_call_output");
      return undefined;
    },
  });

  const [expectedCallId] = toolCall.id.split("|");
  assert.equal(replayedCalls.length, 1, "the function_call must be replayed");
  assert.equal(replayedCalls[0]!.call_id, expectedCallId);
  assert.equal(replayedOutputs.length, 1, "the function_call_output must be replayed");
  assert.equal(replayedOutputs[0]!.call_id, expectedCallId);
  assert.equal(replayedOutputs[0]!.output, "Printed successfully.");
  assert.equal(turn2.stopReason, "stop", "turn 2 must complete normally");
  assert.ok(
    turn2.content.some((b) => b.type === "text" && b.text.length > 0),
    "turn 2 must produce a text reply",
  );
  log("Turn 2 OK: function_call and function_call_output replayed with matching call_id");

  const totalCost = [turn1, turn2].reduce((sum, m) => sum + m.usage.cost.total, 0);
  log(`All assertions passed. Total token cost: $${totalCost.toFixed(4)}`);
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
