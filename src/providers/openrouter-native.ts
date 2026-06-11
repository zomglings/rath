/**
 * "openrouter-native": a pi-ai API provider for OpenRouter's Chat Completions
 * endpoint with OpenRouter's server-side ("hosted") tools — `web_search`
 * (and, opt-in, `web_fetch`).
 *
 * pi-ai routes every OpenRouter model through the stock "openai-completions"
 * api (verified: all openrouter models carry `api: "openai-completions"`).
 * That provider never reads the `url_citation` annotations OpenRouter returns
 * for web search, so the citations are invisible. This provider is a thin
 * conversion layer, built on the official `openai` SDK pointed at OpenRouter's
 * OpenAI-compatible base URL, that:
 *
 *  - sends `{ type: "openrouter:web_search" }` (web search on by default; an
 *    option disables it) and optionally `{ type: "openrouter:web_fetch" }`
 *    alongside any client function tools;
 *  - captures `url_citation` annotations as structured `citations` on text
 *    blocks (extra fields survive structural typing and JSON serialization);
 *  - captures the per-turn server-tool artifact as a `hostedToolCall` block
 *    whose `raw` is OpenRouter's verbatim annotation items.
 *
 * Wire-format note — why there is no replayable tool-call/result item.
 * OpenRouter runs server tools entirely inside one response: it "intercepts
 * the tool call, executes it server-side, and returns the result to the
 * model", then surfaces excerpts to the caller via `url_citation` annotations
 * on the assistant message. The Chat Completions wire format has no
 * client-visible `tool_calls`/`tool` messages for a server tool, and
 * `ChatCompletionAssistantMessageParam` has no `annotations` field
 * (verified against the openai SDK types), so annotations cannot be — and do
 * not need to be — echoed back. The synthesized assistant text already
 * carries the search's information forward across turns, exactly as the stock
 * completions provider replays any assistant message. The captured `raw`
 * items therefore exist for lossless persistence and a faithful
 * hostedToolCall record; replaying them to the wire is intentionally a no-op
 * (see convertNativeMessages). This is the one structural difference from
 * openai-native, whose Responses API does expose replayable `web_search_call`
 * output items.
 *
 * Built on the openai SDK (not hand-rolled fetch): OpenRouter's endpoint is
 * OpenAI-compatible, so the SDK's request/stream plumbing, retries, and types
 * apply directly. The only non-OpenAI piece is the `openrouter:*` tool type,
 * which is added as a structurally-typed extension to the tools array.
 *
 * Only rath's own code switches over the extended block types. Contexts that
 * contain extended blocks must be flattened before being sent to a stock
 * provider (not implemented here).
 */

import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  calculateCost,
  clampThinkingLevel,
  createAssistantMessageEventStream,
  getEnvApiKey,
  getModel,
  type KnownProvider,
  type Model,
  parseStreamingJson,
  registerApiProvider,
  type StopReason,
  type StreamFunction,
  type StreamOptions,
  type ToolCall,
} from "@earendil-works/pi-ai";
import OpenAI from "openai";
import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources/chat/completions/completions.js";
import type { CompletionUsage } from "openai/resources/completions.js";
import {
  type HostedContentBlock,
  type HostedTextContent,
  type HostedToolCallContent,
  isRenderedCitations,
  type UrlCitation,
} from "../hosted-tools.js";

export const OPENROUTER_NATIVE_API = "openrouter-native";

/**
 * OpenRouter server (hosted) tools this provider supports, by the suffix of
 * their `openrouter:<name>` tool type. The full catalogue lives in
 * OpenRouter's server-tools docs; support here is web search (+ fetch), the
 * tools that produce `url_citation` annotations.
 */
export type HostedToolName = "web_search" | "web_fetch";

/**
 * One `url_citation` annotation exactly as OpenRouter returns it on the Chat
 * Completions wire (nested under `url_citation`, with OpenRouter's extra
 * `content` excerpt). Grounded in openai's `ChatCompletionMessage.Annotation`
 * plus OpenRouter's documented `content` field. This is the verbatim raw item
 * the hostedToolCall block preserves.
 */
export interface OpenRouterUrlCitationAnnotation {
  type: "url_citation";
  url_citation: {
    url: string;
    title: string;
    /** Excerpt OpenRouter includes when available; not in the OpenAI shape. */
    content?: string;
    start_index?: number;
    end_index?: number;
  };
}

/**
 * The raw item for an openrouter-native hosted tool call: the verbatim set of
 * `url_citation` annotation items the turn produced. (OpenRouter exposes no
 * other server-tool wire item on Chat Completions; see the file header.)
 */
export interface OpenRouterHostedToolCallItem {
  /** Always "web_search" for v1; "web_fetch" reserved for the fetch tool. */
  toolName: HostedToolName;
  annotations: OpenRouterUrlCitationAnnotation[];
}

/** A hosted tool call block as produced by this provider. */
export type OpenRouterNativeHostedToolCall = HostedToolCallContent<OpenRouterHostedToolCallItem> & {
  toolName: HostedToolName;
  status: "completed";
};

export interface OpenRouterNativeOptions extends StreamOptions {
  reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh";
  /**
   * Hosted web search tool configuration. `true` (default) sends
   * `{ type: "openrouter:web_search" }`; `false` disables it; an object
   * supplies `parameters` (engine, max_results, allowed_domains, ...).
   */
  webSearch?: boolean | Record<string, unknown>;
  /**
   * Hosted web fetch tool. Off by default; `true` sends
   * `{ type: "openrouter:web_fetch" }`; an object supplies `parameters`.
   */
  webFetch?: boolean | Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Model helper
// ---------------------------------------------------------------------------

/**
 * Clone a stock OpenRouter model entry, re-pointing it at the
 * openrouter-native provider. Pass any model id from pi-ai's `openrouter`
 * registry (e.g. "openai/gpt-4o", "anthropic/claude-3.5-sonnet").
 */
export function openrouterNativeModel(modelId: string): Model<typeof OPENROUTER_NATIVE_API> {
  const base = getModel("openrouter" as KnownProvider, modelId as never) as Model<string>;
  if (!base) {
    throw new Error(`Unknown OpenRouter model: ${modelId}`);
  }
  return { ...base, api: OPENROUTER_NATIVE_API } as Model<typeof OPENROUTER_NATIVE_API>;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

let registered = false;

/** Register the openrouter-native provider with pi-ai. Idempotent. */
export function registerOpenRouterNative(): void {
  if (registered) {
    return;
  }
  registerApiProvider({
    api: OPENROUTER_NATIVE_API,
    stream: streamOpenRouterNative,
    streamSimple: (model, context, options) => {
      // Clamp to the model's supported levels so an unsupported request does
      // not 400 every turn. "off" is excluded: clamping it would map to the
      // model's lowest supported level, silently forcing reasoning on against
      // an explicit request to disable it.
      const requested = options?.reasoning as string | undefined;
      const reasoningEffort =
        options?.reasoning && requested !== "off"
          ? clampThinkingLevel(model, options.reasoning)
          : undefined;
      return streamOpenRouterNative(model, context, {
        ...options,
        reasoningEffort: reasoningEffort as OpenRouterNativeOptions["reasoningEffort"],
      });
    },
  });
  registered = true;
}

// ---------------------------------------------------------------------------
// Provider stream function
// ---------------------------------------------------------------------------

export const streamOpenRouterNative: StreamFunction<
  typeof OPENROUTER_NATIVE_API,
  OpenRouterNativeOptions
> = (model, context, options) => {
  const stream = createAssistantMessageEventStream();
  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };
    try {
      const apiKey =
        options?.apiKey || getEnvApiKey(model.provider) || process.env.OPENROUTER_API_KEY;
      if (!apiKey) {
        throw new Error(
          "OpenRouter API key is required. Set OPENROUTER_API_KEY or pass options.apiKey.",
        );
      }
      const headers: Record<string, string> = { ...model.headers };
      Object.assign(headers, options?.headers);
      const client = new OpenAI({ apiKey, baseURL: model.baseUrl, defaultHeaders: headers });

      let params: ChatCompletionCreateParamsStreaming = buildParams(model, context, options);
      const nextParams = await options?.onPayload?.(params, model);
      if (nextParams !== undefined) {
        params = nextParams as ChatCompletionCreateParamsStreaming;
      }
      const requestOptions = {
        ...(options?.signal ? { signal: options.signal } : {}),
        ...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {}),
        ...(options?.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
      };
      const { data: openrouterStream, response } = await client.chat.completions
        .create(params, requestOptions)
        .withResponse();
      await options?.onResponse?.(
        { status: response.status, headers: headersToRecord(response.headers) },
        model,
      );
      stream.push({ type: "start", partial: output });
      await processNativeStream(openrouterStream, output, stream, model);
      if (options?.signal?.aborted) {
        throw new Error("Request was aborted");
      }
      if (output.stopReason === "aborted" || output.stopReason === "error") {
        throw new Error(output.errorMessage || "An unknown error occurred");
      }
      stream.push({ type: "done", reason: output.stopReason, message: output });
      stream.end();
    } catch (error) {
      for (const block of output.content) {
        // partialJson is only a streaming scratch buffer; never persist it.
        delete (block as { partialJson?: string }).partialJson;
      }
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();
  return stream;
};

// ---------------------------------------------------------------------------
// Citation conversion (OpenRouter Chat Completions annotation <-> Citation)
// ---------------------------------------------------------------------------

/**
 * OpenRouter's `url_citation` annotation -> rath `UrlCitation`. Returns
 * undefined for annotation types or malformed items we do not capture. The
 * indices default to 0 when absent so a citation without a span is still a
 * valid structural `UrlCitation` (endIndex >= startIndex preserved).
 */
export function annotationToCitation(annotation: { type?: string }): UrlCitation | undefined {
  if (annotation.type !== "url_citation") {
    return undefined;
  }
  const inner = (annotation as OpenRouterUrlCitationAnnotation).url_citation;
  if (!inner || typeof inner.url !== "string") {
    return undefined;
  }
  return {
    type: "url_citation",
    url: inner.url,
    title: typeof inner.title === "string" ? inner.title : "",
    startIndex: typeof inner.start_index === "number" ? inner.start_index : 0,
    endIndex:
      typeof inner.end_index === "number"
        ? inner.end_index
        : typeof inner.start_index === "number"
          ? inner.start_index
          : 0,
  };
}

// ---------------------------------------------------------------------------
// Context -> Chat Completions params
// ---------------------------------------------------------------------------

/**
 * A server tool entry. Not an OpenAI tool type, so it is modelled separately
 * and unioned into the tools array; OpenRouter recognizes `openrouter:*`.
 */
interface OpenRouterServerTool {
  type: `openrouter:${HostedToolName}`;
  parameters?: Record<string, unknown>;
}

type OpenRouterTool = ChatCompletionTool | OpenRouterServerTool;

export function buildParams(
  model: Model<typeof OPENROUTER_NATIVE_API>,
  context: Context,
  options?: OpenRouterNativeOptions,
): ChatCompletionCreateParamsStreaming {
  const params: ChatCompletionCreateParamsStreaming = {
    model: model.id,
    messages: convertNativeMessages(model, context),
    stream: true,
    stream_options: { include_usage: true },
  };
  if (options?.maxTokens) {
    params.max_tokens = options.maxTokens;
  }
  if (options?.temperature !== undefined) {
    params.temperature = options.temperature;
  }

  const tools: OpenRouterTool[] = [];
  if (options?.webSearch !== false) {
    const tool: OpenRouterServerTool = { type: "openrouter:web_search" };
    if (typeof options?.webSearch === "object") {
      tool.parameters = options.webSearch;
    }
    tools.push(tool);
  }
  if (options?.webFetch) {
    const tool: OpenRouterServerTool = { type: "openrouter:web_fetch" };
    if (typeof options.webFetch === "object") {
      tool.parameters = options.webFetch;
    }
    tools.push(tool);
  }
  for (const tool of context.tools ?? []) {
    tools.push({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters as Record<string, unknown>,
      },
    });
  }
  if (tools.length > 0) {
    // The server-tool entries are structurally-typed extensions, not OpenAI
    // `ChatCompletionTool`s; cast at the single boundary where they cross into
    // the SDK params.
    params.tools = tools as ChatCompletionTool[];
  }

  if (model.reasoning) {
    // OpenRouter normalizes reasoning across providers via a nested reasoning
    // object on the request. Map rath's effort through the model's level map;
    // with no effort requested, fall back to the model's "off" level (skipped
    // when that level is null, i.e. reasoning cannot be turned off).
    const effort = options?.reasoningEffort
      ? (model.thinkingLevelMap?.[options.reasoningEffort] ?? options.reasoningEffort)
      : model.thinkingLevelMap?.off;
    if (effort !== null && effort !== undefined) {
      (params as { reasoning?: { effort: string } }).reasoning = { effort: String(effort) };
    }
  }
  return params;
}

export function convertNativeMessages(
  model: Model<typeof OPENROUTER_NATIVE_API>,
  context: Context,
): ChatCompletionMessageParam[] {
  const items: ChatCompletionMessageParam[] = [];
  if (context.systemPrompt) {
    items.push({
      role: model.reasoning ? "developer" : "system",
      content: context.systemPrompt,
    });
  }
  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        items.push({ role: "user", content: msg.content });
      } else {
        const content = msg.content.map((item) =>
          item.type === "text"
            ? ({ type: "text", text: item.text } as const)
            : ({
                type: "image_url",
                image_url: { url: `data:${item.mimeType};base64,${item.data}` },
              } as const),
        );
        if (content.length > 0) {
          items.push({ role: "user", content });
        }
      }
    } else if (msg.role === "assistant") {
      const toolCalls: NonNullable<ChatCompletionAssistantMessageParam["tool_calls"]> = [];
      const reasoningDetails: unknown[] = [];
      let text = "";
      for (const block of msg.content as HostedContentBlock[]) {
        if (block.type === "text") {
          // The rendered-citations trailer is a display/persistence artifact;
          // never replay it (it would duplicate text the model never produced).
          if (isRenderedCitations(block)) {
            continue;
          }
          text += block.text;
        } else if (block.type === "toolCall") {
          const [callId] = block.id.split("|");
          toolCalls.push({
            id: callId ?? block.id,
            type: "function",
            function: { name: block.name, arguments: JSON.stringify(block.arguments) },
          });
          if (block.thoughtSignature) {
            try {
              reasoningDetails.push(JSON.parse(block.thoughtSignature));
            } catch {
              // A hand-edited or corrupted signature must not crash the turn.
            }
          }
        }
        // thinking and hostedToolCall blocks are intentionally not replayed to
        // the wire. Thinking blocks have no Chat Completions assistant-message
        // slot here; their encrypted-reasoning continuation, when an upstream
        // requires it, rides on tool calls via reasoning_details above.
        // hostedToolCall raw items are output-only annotations OpenRouter does
        // not accept on a request (see the file header).
      }
      const assistantMsg: ChatCompletionAssistantMessageParam = { role: "assistant" };
      if (text.length > 0) {
        assistantMsg.content = text;
      }
      if (toolCalls.length > 0) {
        assistantMsg.tool_calls = toolCalls;
      }
      if (reasoningDetails.length > 0) {
        (assistantMsg as { reasoning_details?: unknown[] }).reasoning_details = reasoningDetails;
      }
      // Skip empty assistant turns (e.g. an errored turn that produced nothing):
      // the API rejects an assistant message with neither content nor tool_calls.
      if (assistantMsg.content !== undefined || assistantMsg.tool_calls !== undefined) {
        items.push(assistantMsg);
      }
    } else if (msg.role === "toolResult") {
      const textResult = msg.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      const [callId] = msg.toolCallId.split("|");
      items.push({
        role: "tool",
        tool_call_id: callId ?? msg.toolCallId,
        content: textResult.length > 0 ? textResult : "(see attached image)",
      });
    }
  }
  return items;
}

// ---------------------------------------------------------------------------
// Chat Completions stream -> pi-ai events
// ---------------------------------------------------------------------------

/** One entry of OpenRouter's `reasoning_details` array (its reasoning wire). */
interface OpenRouterReasoningDetail {
  type?: string;
  id?: string;
  data?: string;
  [key: string]: unknown;
}

/** A streaming delta, widened for OpenRouter's non-standard extra fields. */
interface OpenRouterDelta extends ChatCompletionChunk.Choice.Delta {
  annotations?: OpenRouterUrlCitationAnnotation[];
  reasoning?: string | null;
  reasoning_content?: string | null;
  reasoning_details?: OpenRouterReasoningDetail[];
}

async function processNativeStream(
  openrouterStream: AsyncIterable<ChatCompletionChunk>,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  model: Model<typeof OPENROUTER_NATIVE_API>,
): Promise<void> {
  const blocks = output.content as HostedContentBlock[];
  let textBlock: HostedTextContent | null = null;
  let thinkingIndex: number | null = null;
  const toolCallsByIndex = new Map<number, ToolCall & { partialJson?: string }>();
  // Accumulate annotations across chunks; OpenRouter may stream them
  // incrementally and/or repeat the full set on the final delta. Dedupe by
  // (url, start, end) so a repeated final set does not double-count.
  const annotations: OpenRouterUrlCitationAnnotation[] = [];
  const seenAnnotations = new Set<string>();
  let hasFinishReason = false;

  const indexOf = (block: HostedContentBlock) => blocks.indexOf(block);

  const ensureTextBlock = (): HostedTextContent => {
    if (!textBlock) {
      textBlock = { type: "text", text: "", citations: [] } as HostedTextContent;
      blocks.push(textBlock);
      stream.push({ type: "text_start", contentIndex: indexOf(textBlock), partial: output });
    }
    return textBlock;
  };

  for await (const chunk of openrouterStream) {
    if (!chunk || typeof chunk !== "object") {
      continue;
    }
    output.responseId ||= chunk.id;
    // OpenRouter can surface an in-band error as a top-level `error` field on a
    // chunk (e.g. upstream provider failure) instead of an HTTP error the SDK
    // would throw. Treat it as a hard failure.
    const chunkError = (chunk as { error?: { message?: string; code?: string | number } }).error;
    if (chunkError) {
      throw new Error(
        `OpenRouter error${chunkError.code !== undefined ? ` ${chunkError.code}` : ""}: ${
          chunkError.message || "unknown error"
        }`,
      );
    }
    if (chunk.usage) {
      finalizeUsage(chunk.usage, output, model);
    }
    const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : undefined;
    if (!choice) {
      continue;
    }
    if (choice.finish_reason) {
      output.stopReason = mapStopReason(choice.finish_reason);
      if (output.stopReason === "error") {
        output.errorMessage = `Provider finish_reason: ${choice.finish_reason}`;
      }
      hasFinishReason = true;
    }
    const delta = choice.delta as OpenRouterDelta | undefined;
    if (!delta) {
      continue;
    }

    if (typeof delta.content === "string" && delta.content.length > 0) {
      const block = ensureTextBlock();
      block.text += delta.content;
      stream.push({
        type: "text_delta",
        contentIndex: indexOf(block),
        delta: delta.content,
        partial: output,
      });
    }

    const reasoningDelta =
      typeof delta.reasoning === "string" && delta.reasoning.length > 0
        ? delta.reasoning
        : typeof delta.reasoning_content === "string" && delta.reasoning_content.length > 0
          ? delta.reasoning_content
          : undefined;
    if (reasoningDelta) {
      if (thinkingIndex === null) {
        blocks.push({ type: "thinking", thinking: "" });
        thinkingIndex = blocks.length - 1;
        stream.push({ type: "thinking_start", contentIndex: thinkingIndex, partial: output });
      }
      const block = blocks[thinkingIndex] as HostedContentBlock & { type: "thinking" };
      block.thinking += reasoningDelta;
      stream.push({
        type: "thinking_delta",
        contentIndex: thinkingIndex,
        delta: reasoningDelta,
        partial: output,
      });
    }

    if (Array.isArray(delta.annotations)) {
      for (const annotation of delta.annotations) {
        if (annotation?.type !== "url_citation" || !annotation.url_citation?.url) {
          continue;
        }
        const inner = annotation.url_citation;
        const key = `${inner.url} ${inner.start_index ?? ""} ${inner.end_index ?? ""}`;
        if (seenAnnotations.has(key)) {
          continue;
        }
        seenAnnotations.add(key);
        annotations.push(annotation);
        const citation = annotationToCitation(annotation);
        if (citation) {
          const block = ensureTextBlock();
          block.citations = block.citations ?? [];
          block.citations.push(citation);
        }
      }
    }

    if (Array.isArray(delta.tool_calls)) {
      for (const toolCall of delta.tool_calls) {
        const streamIndex = typeof toolCall.index === "number" ? toolCall.index : 0;
        let block = toolCallsByIndex.get(streamIndex);
        if (!block) {
          block = {
            type: "toolCall",
            id: toolCall.id || "",
            name: toolCall.function?.name || "",
            arguments: {},
            partialJson: "",
          } as ToolCall & { partialJson?: string };
          toolCallsByIndex.set(streamIndex, block);
          blocks.push(block);
          stream.push({ type: "toolcall_start", contentIndex: indexOf(block), partial: output });
        }
        if (!block.id && toolCall.id) {
          block.id = toolCall.id;
        }
        if (!block.name && toolCall.function?.name) {
          block.name = toolCall.function.name;
        }
        if (toolCall.function?.arguments) {
          block.partialJson = (block.partialJson ?? "") + toolCall.function.arguments;
          block.arguments = parseStreamingJson(block.partialJson);
          stream.push({
            type: "toolcall_delta",
            contentIndex: indexOf(block),
            delta: toolCall.function.arguments,
            partial: output,
          });
        }
      }
    }

    // OpenRouter carries per-provider encrypted reasoning in reasoning_details,
    // keyed to a tool call id. Some upstreams (Anthropic, DeepSeek via
    // OpenRouter) reject a replayed tool call whose paired reasoning is missing,
    // so stash the verbatim detail on the matching tool call's thoughtSignature
    // to replay it (mirrors pi-ai's stock openai-completions handling).
    if (Array.isArray(delta.reasoning_details)) {
      for (const detail of delta.reasoning_details) {
        if (detail?.type === "reasoning.encrypted" && detail.id && detail.data) {
          const target = blocks.find(
            (b): b is ToolCall => b.type === "toolCall" && b.id === detail.id,
          );
          if (target) {
            target.thoughtSignature = JSON.stringify(detail);
          }
        }
      }
    }
  }

  // Finalize text and emit its end event.
  if (textBlock) {
    const block: HostedTextContent = textBlock;
    stream.push({
      type: "text_end",
      contentIndex: indexOf(block),
      content: block.text,
      partial: output,
    });
  }
  if (thinkingIndex !== null) {
    const block = blocks[thinkingIndex] as HostedContentBlock & { type: "thinking" };
    stream.push({
      type: "thinking_end",
      contentIndex: thinkingIndex,
      content: block.thinking,
      partial: output,
    });
  }
  for (const block of toolCallsByIndex.values()) {
    block.arguments = parseStreamingJson(block.partialJson || "{}");
    delete block.partialJson;
    stream.push({
      type: "toolcall_end",
      contentIndex: indexOf(block),
      toolCall: block,
      partial: output,
    });
  }

  // Record the web-search server tool call as a hostedToolCall block carrying
  // OpenRouter's verbatim annotation items. Placed last so it does not perturb
  // the text/citation block indices the stream events referenced.
  if (annotations.length > 0) {
    const hosted: OpenRouterNativeHostedToolCall = {
      type: "hostedToolCall",
      id: output.responseId ? `${output.responseId}:web_search` : "web_search",
      toolName: "web_search",
      status: "completed",
      raw: { toolName: "web_search", annotations },
    };
    blocks.push(hosted);
  }

  if (output.stopReason === "stop" && blocks.some((b) => b.type === "toolCall")) {
    output.stopReason = "toolUse";
  }
  if (!hasFinishReason && output.stopReason === "stop") {
    // A stream that ended without a finish_reason is truncated; report length
    // rather than a clean stop so callers do not treat it as complete.
    output.stopReason = "length";
  }
}

/** Capture token usage and cost from a streaming usage chunk. */
function finalizeUsage(
  usage: CompletionUsage,
  output: AssistantMessage,
  model: Model<typeof OPENROUTER_NATIVE_API>,
): void {
  const promptTokens = usage.prompt_tokens || 0;
  const cacheRead = usage.prompt_tokens_details?.cached_tokens || 0;
  output.usage = {
    input: Math.max(0, promptTokens - cacheRead),
    output: usage.completion_tokens || 0,
    cacheRead,
    cacheWrite: 0,
    totalTokens: usage.total_tokens || 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
  calculateCost(model, output.usage);
}

function mapStopReason(reason: ChatCompletionChunk.Choice["finish_reason"]): StopReason {
  switch (reason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
    case "function_call":
      return "toolUse";
    case "content_filter":
      return "error";
    default:
      return "stop";
  }
}

function headersToRecord(headers: Headers): Record<string, string> {
  const record: Record<string, string> = {};
  headers.forEach((value, key) => {
    record[key] = value;
  });
  return record;
}
