/**
 * Acceptance spike for the openai-native provider (issue #5).
 *
 * Runs a 3-turn conversation through pi-ai's stream() with the
 * "openai-native" provider and asserts:
 *  1. Turn 1 triggers a hosted web search and the assistant message carries
 *     structured citations (URL, title, text span).
 *  2. Turn 2 references turn 1's results; raw hosted-tool items are replayed
 *     to the API without loss.
 *  3. The full context survives JSON serialize/deserialize (session
 *     persistence) with citations intact; turn 3 runs on the deserialized
 *     context.
 *
 * Requires OPENAI_API_KEY. Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import { type AssistantMessage, type Context, stream } from "@earendil-works/pi-ai";
import {
  contentBlocks,
  getCitations,
  getHostedToolCalls,
  isHostedToolCall,
  type OpenAINativeOptions,
  openaiNativeModel,
  registerOpenAINative,
} from "../index.js";

const MODEL_ID = process.env.RATH_TEST_MODEL || "gpt-5-mini";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function runTurn(context: Context, options?: OpenAINativeOptions): Promise<AssistantMessage> {
  const model = openaiNativeModel(MODEL_ID);
  const events = stream(model, context, { reasoningEffort: "low", ...options });
  const message = await events.result();
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    throw new Error(`Turn failed (${message.stopReason}): ${message.errorMessage}`);
  }
  return message;
}

function describeMessage(message: AssistantMessage): string {
  return contentBlocks(message)
    .map((block) => {
      if (isHostedToolCall(block)) {
        return `hostedToolCall(${block.toolName}, ${block.status})`;
      }
      if (block.type === "text") {
        return `text(${block.text.length} chars, ${getCitations(block).length} citations)`;
      }
      return block.type;
    })
    .join(", ");
}

async function main(): Promise<void> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  registerOpenAINative();

  const context: Context = {
    systemPrompt:
      "You are a research assistant. When asked about current events or " +
      "facts that change over time, use web search and cite your sources.",
    messages: [],
  };

  // ---- Turn 1: trigger a hosted web search, expect structured citations ----
  log(`Turn 1 (model: ${MODEL_ID}): asking a question that requires web search`);
  context.messages.push({
    role: "user",
    content:
      "What is the current latest LTS version of Node.js? Search the web and cite your sources.",
    timestamp: Date.now(),
  });
  const turn1 = await runTurn(context);
  log(`Turn 1 content: ${describeMessage(turn1)}`);

  const searches1 = getHostedToolCalls(turn1);
  assert.ok(searches1.length > 0, "turn 1 must contain at least one hosted web_search_call block");
  for (const search of searches1) {
    assert.equal(search.raw.type, "web_search_call", "raw item must be a web_search_call");
    assert.equal(search.status, "completed", "web search call must complete");
  }

  const citations1 = turn1.content.filter((b) => b.type === "text").flatMap((b) => getCitations(b));
  assert.ok(citations1.length > 0, "turn 1 text must carry at least one citation");
  for (const citation of citations1) {
    assert.ok(citation.type === "url_citation", "web search must produce url_citations");
    assert.ok(citation.url.startsWith("http"), `citation has a URL (got: ${citation.url})`);
    assert.equal(typeof citation.title, "string", "citation has a title");
    assert.ok(
      Number.isInteger(citation.startIndex) &&
        Number.isInteger(citation.endIndex) &&
        citation.endIndex >= citation.startIndex,
      "citation has a valid text span",
    );
  }
  log(`Turn 1 OK: ${searches1.length} search call(s), ${citations1.length} citation(s)`);
  context.messages.push(turn1);

  // ---- Turn 2: replay turn 1's raw hosted-tool items without loss ----
  log("Turn 2: referencing turn 1's results; verifying raw item replay");
  context.messages.push({
    role: "user",
    content:
      "Based only on the search you already performed, list the source URLs you used. Reply briefly.",
    timestamp: Date.now(),
  });
  let replayedItems: unknown[] = [];
  const turn2 = await runTurn(context, {
    onPayload: (payload) => {
      const params = payload as { input: { type?: string }[] };
      replayedItems = params.input.filter((item) => item.type === "web_search_call");
      return undefined;
    },
  });
  log(`Turn 2 content: ${describeMessage(turn2)}`);

  assert.equal(
    replayedItems.length,
    searches1.length,
    "every turn-1 web_search_call must be replayed in the turn-2 request",
  );
  assert.deepEqual(
    replayedItems,
    searches1.map((s) => s.raw),
    "replayed web_search_call items must match the captured raw items exactly",
  );
  assert.ok(
    turn2.content.some((b) => b.type === "text" && b.text.length > 0),
    "turn 2 must produce a text reply",
  );
  log(`Turn 2 OK: ${replayedItems.length} raw item(s) replayed without loss`);
  context.messages.push(turn2);

  // ---- Turn 3: context survives JSON serialize/deserialize ----
  log("Turn 3: JSON round-trip of the full context, then one more turn");
  const serialized = JSON.stringify(context);
  const restored = JSON.parse(serialized) as Context;
  assert.deepEqual(restored, context, "context must survive JSON serialize/deserialize");

  const restoredCitations = restored.messages
    .filter((m): m is AssistantMessage => m.role === "assistant")
    .flatMap((m) => m.content.filter((b) => b.type === "text").flatMap((b) => getCitations(b)));
  assert.deepEqual(restoredCitations, citations1, "citations must survive the JSON round-trip");

  restored.messages.push({
    role: "user",
    content: "In one short sentence: what version did you report in your first answer?",
    timestamp: Date.now(),
  });
  let replayedItems3: unknown[] = [];
  const turn3 = await runTurn(restored, {
    onPayload: (payload) => {
      const params = payload as { input: { type?: string }[] };
      replayedItems3 = params.input.filter((item) => item.type === "web_search_call");
      return undefined;
    },
  });
  log(`Turn 3 content: ${describeMessage(turn3)}`);

  assert.deepEqual(
    replayedItems3,
    searches1.map((s) => s.raw),
    "raw items must replay identically from the deserialized context",
  );
  assert.ok(
    turn3.content.some((b) => b.type === "text" && b.text.length > 0),
    "turn 3 must produce a text reply",
  );
  log("Turn 3 OK: deserialized context replayed successfully");

  const totalCost = [turn1, turn2, turn3].reduce((sum, m) => sum + m.usage.cost.total, 0);
  log(`All assertions passed. Total token cost: $${totalCost.toFixed(4)} (excludes search fees)`);
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
