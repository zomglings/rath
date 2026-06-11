/**
 * Integration test: the openai-native provider drives pi's agent loop
 * (@earendil-works/pi-agent-core) with hosted tools disabled.
 *
 * Verifies that a request made through openai-native with no server-side
 * tools enabled is processed by the stock agent loop end to end:
 *  1. The model calls a client-side tool; the loop executes it and feeds the
 *     result back.
 *  2. No hosted tool is sent in any request payload and no hostedToolCall
 *     blocks appear in the transcript.
 *  3. The final assistant message uses the tool result.
 *
 * Requires OPENAI_API_KEY. Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import { type AgentContext, type AgentTool, agentLoop } from "@earendil-works/pi-agent-core";
import { type Message, type SimpleStreamOptions, streamSimple, Type } from "@earendil-works/pi-ai";
import { getHostedToolCalls, openaiNativeModel, registerOpenAINative } from "../index.js";

const MODEL_ID = process.env.RATH_TEST_MODEL || "gpt-5.5";
const CODEWORD = "PERIWINKLE-42";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  registerOpenAINative();
  const model = openaiNativeModel(MODEL_ID);

  let toolExecutions = 0;
  const lookupCodeword: AgentTool = {
    name: "lookup_codeword",
    label: "Lookup codeword",
    description: "Returns the secret codeword. Takes no arguments.",
    parameters: Type.Object({}),
    execute: async () => {
      toolExecutions++;
      return {
        content: [{ type: "text", text: `The codeword is ${CODEWORD}.` }],
        details: null,
      };
    },
  };

  const context: AgentContext = {
    systemPrompt:
      "You are a terse assistant. Use the lookup_codeword tool when asked for the codeword.",
    messages: [],
    tools: [lookupCodeword],
  };

  // Capture every request payload to prove no hosted tool is ever sent.
  const payloadTools: { type?: string; name?: string }[][] = [];
  const streamFn: typeof streamSimple = (m, ctx, options) =>
    streamSimple(m, ctx, {
      ...options,
      webSearch: false,
      onPayload: (payload: unknown) => {
        const params = payload as { tools?: { type?: string; name?: string }[] };
        payloadTools.push(params.tools ?? []);
        return undefined;
      },
    } as SimpleStreamOptions);

  log(`Running agent loop (model: ${MODEL_ID}) with hosted tools disabled`);
  const events = agentLoop(
    [
      {
        role: "user",
        content: "What is the codeword? Look it up and reply with exactly the codeword.",
        timestamp: Date.now(),
      },
    ],
    context,
    {
      model,
      reasoning: "low",
      convertToLlm: (messages) =>
        messages.filter(
          (m): m is Message =>
            m.role === "user" || m.role === "assistant" || m.role === "toolResult",
        ),
    },
    undefined,
    streamFn,
  );
  const newMessages = await events.result();

  const assistants = newMessages.filter((m) => m.role === "assistant");
  const toolResults = newMessages.filter((m) => m.role === "toolResult");
  log(
    `Loop produced ${newMessages.length} message(s): ` +
      `${assistants.length} assistant, ${toolResults.length} toolResult`,
  );

  // 1. The loop executed the client-side tool and fed the result back.
  assert.equal(toolExecutions, 1, "lookup_codeword must execute exactly once");
  assert.equal(toolResults.length, 1, "the loop must produce one toolResult message");
  assert.equal(toolResults[0]!.toolName, "lookup_codeword");
  assert.ok(
    assistants.some((m) => m.content.some((b) => b.type === "toolCall")),
    "an assistant message must carry the toolCall block",
  );

  // 2. No hosted tools requested, none returned.
  assert.ok(payloadTools.length >= 2, "the loop must make at least two LLM requests");
  for (const tools of payloadTools) {
    assert.ok(
      tools.every((t) => t.type === "function"),
      `request payloads must contain only function tools (got: ${tools.map((t) => t.type).join(", ")})`,
    );
    assert.ok(
      tools.some((t) => t.name === "lookup_codeword"),
      "the client tool must be present in every request payload",
    );
  }
  for (const message of assistants) {
    assert.equal(getHostedToolCalls(message).length, 0, "no hostedToolCall blocks expected");
  }

  // 3. The final answer uses the tool result.
  const finalText = assistants[assistants.length - 1]!.content.filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
  assert.ok(
    finalText.includes(CODEWORD),
    `final reply must contain ${CODEWORD} (got: ${finalText})`,
  );

  const totalCost = assistants.reduce((sum, m) => sum + m.usage.cost.total, 0);
  log(`All assertions passed. Total token cost: $${totalCost.toFixed(4)}`);
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
