/**
 * Acceptance spike for the openrouter-native provider (issue #12).
 *
 * Runs a 3-turn conversation through pi-ai's stream() with the
 * "openrouter-native" provider and asserts:
 *  1. Turn 1 triggers OpenRouter's hosted web search and the assistant message
 *     carries structured citations (URL, title, text span) plus a
 *     hostedToolCall block whose raw item holds OpenRouter's verbatim
 *     url_citation annotations.
 *  2. Turn 2 references turn 1's results; the captured raw hosted-tool item
 *     survives onto the next request's context without loss.
 *  3. The full context survives JSON serialize/deserialize (session
 *     persistence) with citations and raw items intact; turn 3 runs on the
 *     deserialized context.
 *
 * Requires OPENROUTER_API_KEY. Exits 0 on success, 1 on failure.
 *
 * Note: unlike openai-native (Responses API), OpenRouter executes the web
 * search server-side inside one response and surfaces results only as
 * annotations — there is no replayable web_search_call wire item, and
 * Chat Completions assistant messages carry no annotations field. So replay
 * is verified at the captured-block level (the raw item round-trips
 * losslessly), not as an echoed request item.
 */
import assert from "node:assert/strict";
import { type AssistantMessage, type Context, stream } from "@earendil-works/pi-ai";
import {
  contentBlocks,
  getCitations,
  getHostedToolCalls,
  isHostedToolCall,
  type OpenRouterHostedToolCallItem,
  type OpenRouterNativeOptions,
  openrouterNativeModel,
  registerOpenRouterNative,
} from "../index.js";

const MODEL_ID = process.env.RATH_TEST_MODEL || "openai/gpt-5.5";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function runTurn(
  context: Context,
  options?: OpenRouterNativeOptions,
): Promise<AssistantMessage> {
  const model = openrouterNativeModel(MODEL_ID);
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
  if (!process.env.OPENROUTER_API_KEY) {
    throw new Error("OPENROUTER_API_KEY is not set");
  }
  registerOpenRouterNative();

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

  const searches1 = getHostedToolCalls<OpenRouterHostedToolCallItem>(turn1);
  assert.ok(searches1.length > 0, "turn 1 must contain at least one hosted web_search block");
  for (const search of searches1) {
    assert.equal(search.toolName, "web_search", "hosted tool must be web_search");
    assert.equal(search.status, "completed", "web search call must complete");
    assert.ok(
      Array.isArray(search.raw.annotations) && search.raw.annotations.length > 0,
      "raw item must carry OpenRouter's url_citation annotations",
    );
    for (const annotation of search.raw.annotations) {
      assert.equal(annotation.type, "url_citation", "raw annotation must be a url_citation");
      assert.ok(
        typeof annotation.url_citation?.url === "string",
        "raw annotation must carry a url",
      );
    }
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

  // ---- Turn 2: a follow-up turn; the captured raw item must persist intact ----
  log("Turn 2: referencing turn 1's results; verifying captured raw item integrity");
  const rawBefore = JSON.stringify(searches1.map((s) => s.raw));
  context.messages.push({
    role: "user",
    content:
      "Based on the search you already performed, list the source URLs you used. Reply briefly.",
    timestamp: Date.now(),
  });
  const turn2 = await runTurn(context);
  log(`Turn 2 content: ${describeMessage(turn2)}`);

  // The turn-1 message object is shared in context; its raw items must be
  // untouched by replay conversion (no mutation of captured server-tool data).
  const rawAfter = JSON.stringify(
    getHostedToolCalls<OpenRouterHostedToolCallItem>(turn1).map((s) => s.raw),
  );
  assert.equal(rawAfter, rawBefore, "captured raw items must not mutate across a later turn");
  assert.ok(
    turn2.content.some((b) => b.type === "text" && b.text.length > 0),
    "turn 2 must produce a text reply",
  );
  log("Turn 2 OK: captured raw items survived a later turn unchanged");
  context.messages.push(turn2);

  // ---- Turn 3: context survives JSON serialize/deserialize ----
  log("Turn 3: JSON round-trip of the full context, then one more turn");
  const serialized = JSON.stringify(context);
  const restored = JSON.parse(serialized) as Context;
  assert.deepEqual(restored, context, "context must survive JSON serialize/deserialize");

  const restoredSearches = restored.messages
    .filter((m): m is AssistantMessage => m.role === "assistant")
    .flatMap((m) => getHostedToolCalls<OpenRouterHostedToolCallItem>(m));
  assert.deepEqual(
    restoredSearches.map((s) => s.raw),
    searches1.map((s) => s.raw),
    "raw hosted-tool items must survive the JSON round-trip",
  );

  const restoredCitations = restored.messages
    .filter((m): m is AssistantMessage => m.role === "assistant")
    .flatMap((m) => m.content.filter((b) => b.type === "text").flatMap((b) => getCitations(b)));
  assert.deepEqual(restoredCitations, citations1, "citations must survive the JSON round-trip");

  restored.messages.push({
    role: "user",
    content: "In one short sentence: what version did you report in your first answer?",
    timestamp: Date.now(),
  });
  const turn3 = await runTurn(restored);
  log(`Turn 3 content: ${describeMessage(turn3)}`);
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
