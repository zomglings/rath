/**
 * "openai-native": a pi-ai API provider for the OpenAI Responses API with
 * native (server-side) web search.
 *
 * pi-ai's stock "openai-responses" provider discards `url_citation`
 * annotations (it hardcodes `annotations: []` on replay) and only models
 * client-side function tools. This provider is a conversion layer built on
 * the official `openai` SDK that:
 *
 *  - sends the hosted `web_search` tool alongside any client function tools;
 *  - captures `url_citation` annotations as structured `citations` on text
 *    blocks (extra fields survive structural typing and JSON serialization);
 *  - captures raw `web_search_call` output items as `hostedToolCall` blocks
 *    and replays them verbatim on later turns.
 *
 * Only rath's own code switches over the extended block types. Contexts that
 * contain extended blocks must be flattened before being sent to a stock
 * provider (not implemented here; see issue #5 "out of scope").
 */
import OpenAI from "openai";
import type {
  ResponseCreateParamsStreaming,
  ResponseFunctionWebSearch,
  ResponseInput,
  ResponseInputItem,
  ResponseOutputText,
  ResponseReasoningItem,
  ResponseStreamEvent,
  Tool as OpenAITool,
  WebSearchTool,
} from "openai/resources/responses/responses.js";
import {
  calculateCost,
  createAssistantMessageEventStream,
  getEnvApiKey,
  getModel,
  parseStreamingJson,
  registerApiProvider,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type KnownProvider,
  type Model,
  type StopReason,
  type StreamFunction,
  type StreamOptions,
  type TextContent,
  type ThinkingContent,
  type ToolCall,
} from "@earendil-works/pi-ai";

export const OPENAI_NATIVE_API = "openai-native";

/** Structured citation extracted from a `url_citation` annotation. */
export interface UrlCitation {
  type: "url_citation";
  url: string;
  title: string;
  /** Index of the first character of the cited span in the text block. */
  startIndex: number;
  /** Index one past the last character of the cited span in the text block. */
  endIndex: number;
}

/** Text block extended with citations. Structurally still a TextContent. */
export interface HostedTextContent extends TextContent {
  citations?: UrlCitation[];
}

/**
 * A server-side tool call (e.g. web search) executed by OpenAI. `raw` is the
 * verbatim Responses API output item; it is replayed to the API on later
 * turns. This block type is not part of pi-ai's content union: only
 * openai-native (and rath code) understands it.
 */
export interface HostedToolCallContent {
  type: "hostedToolCall";
  id: string;
  toolName: "web_search";
  status: ResponseFunctionWebSearch["status"];
  raw: ResponseFunctionWebSearch;
}

export type HostedContentBlock = TextContent | ThinkingContent | ToolCall | HostedToolCallContent;

export function isHostedToolCall(block: { type: string }): block is HostedToolCallContent {
  return block.type === "hostedToolCall";
}

/**
 * View an assistant message's content as the extended block union. pi-ai
 * types content as (TextContent | ThinkingContent | ToolCall)[]; messages
 * produced by openai-native additionally carry HostedToolCallContent blocks.
 */
export function contentBlocks(message: AssistantMessage): HostedContentBlock[] {
  return message.content as unknown as HostedContentBlock[];
}

/** Hosted (server-side) tool call blocks on an assistant message. */
export function getHostedToolCalls(message: AssistantMessage): HostedToolCallContent[] {
  return contentBlocks(message).filter(isHostedToolCall);
}

/** Citations on a text block, if any. */
export function getCitations(block: TextContent): UrlCitation[] {
  return (block as HostedTextContent).citations ?? [];
}

export interface OpenAINativeOptions extends StreamOptions {
  reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh";
  reasoningSummary?: "auto" | "detailed" | "concise" | null;
  /**
   * Hosted web search tool configuration. `true` (default) sends
   * `{type: "web_search"}`; `false` disables it; an object customizes
   * filters/user_location.
   */
  webSearch?: boolean | Omit<WebSearchTool, "type">;
}

// ---------------------------------------------------------------------------
// Model helper
// ---------------------------------------------------------------------------

/**
 * Clone a stock OpenAI model entry, re-pointing it at the openai-native
 * provider. Pass any model id from pi-ai's `openai` registry.
 */
export function openaiNativeModel(modelId: string): Model<typeof OPENAI_NATIVE_API> {
  const base = getModel("openai" as KnownProvider, modelId as never) as Model<string>;
  if (!base) {
    throw new Error(`Unknown OpenAI model: ${modelId}`);
  }
  return { ...base, api: OPENAI_NATIVE_API } as Model<typeof OPENAI_NATIVE_API>;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

let registered = false;

/** Register the openai-native provider with pi-ai. Idempotent. */
export function registerOpenAINative(): void {
  if (registered) {
    return;
  }
  registerApiProvider({
    api: OPENAI_NATIVE_API,
    stream: streamOpenAINative,
    streamSimple: (model, context, options) =>
      streamOpenAINative(model, context, {
        ...options,
        reasoningEffort: options?.reasoning,
      }),
  });
  registered = true;
}

// ---------------------------------------------------------------------------
// Provider stream function
// ---------------------------------------------------------------------------

export const streamOpenAINative: StreamFunction<typeof OPENAI_NATIVE_API, OpenAINativeOptions> = (
  model,
  context,
  options,
) => {
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
      const apiKey = options?.apiKey || getEnvApiKey(model.provider) || process.env.OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error(
          "OpenAI API key is required. Set OPENAI_API_KEY or pass options.apiKey.",
        );
      }
      const headers: Record<string, string> = { ...model.headers };
      if (options?.sessionId) {
        headers.session_id = options.sessionId;
        headers["x-client-request-id"] = options.sessionId;
      }
      Object.assign(headers, options?.headers);
      const client = new OpenAI({ apiKey, baseURL: model.baseUrl, defaultHeaders: headers });

      let params: ResponseCreateParamsStreaming = buildParams(model, context, options);
      const nextParams = await options?.onPayload?.(params, model);
      if (nextParams !== undefined) {
        params = nextParams as ResponseCreateParamsStreaming;
      }
      const requestOptions = {
        ...(options?.signal ? { signal: options.signal } : {}),
        ...(options?.timeoutMs !== undefined ? { timeout: options.timeoutMs } : {}),
        ...(options?.maxRetries !== undefined ? { maxRetries: options.maxRetries } : {}),
      };
      const { data: openaiStream, response } = await client.responses
        .create(params, requestOptions)
        .withResponse();
      await options?.onResponse?.(
        { status: response.status, headers: headersToRecord(response.headers) },
        model,
      );
      stream.push({ type: "start", partial: output });
      await processNativeStream(openaiStream, output, stream, model);
      if (options?.signal?.aborted) {
        throw new Error("Request was aborted");
      }
      if (output.stopReason === "aborted" || output.stopReason === "error") {
        throw new Error("An unknown error occurred");
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
// Context -> Responses API params
// ---------------------------------------------------------------------------

function buildParams(
  model: Model<typeof OPENAI_NATIVE_API>,
  context: Context,
  options?: OpenAINativeOptions,
): ResponseCreateParamsStreaming {
  const params: ResponseCreateParamsStreaming = {
    model: model.id,
    input: convertNativeMessages(model, context),
    stream: true,
    store: false,
    include: ["web_search_call.action.sources"],
    prompt_cache_key: options?.sessionId,
  };
  if (options?.maxTokens) {
    params.max_output_tokens = options.maxTokens;
  }
  if (options?.temperature !== undefined) {
    params.temperature = options.temperature;
  }

  const tools: OpenAITool[] = [];
  if (options?.webSearch !== false) {
    const config = typeof options?.webSearch === "object" ? options.webSearch : {};
    tools.push({ type: "web_search", ...config });
  }
  for (const tool of context.tools ?? []) {
    tools.push({
      type: "function",
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters as Record<string, unknown>,
      strict: false,
    });
  }
  if (tools.length > 0) {
    params.tools = tools;
  }

  if (model.reasoning) {
    if (options?.reasoningEffort || options?.reasoningSummary) {
      const effort = options?.reasoningEffort
        ? (model.thinkingLevelMap?.[options.reasoningEffort] ?? options.reasoningEffort)
        : "medium";
      params.reasoning = {
        effort: effort as NonNullable<ResponseCreateParamsStreaming["reasoning"]>["effort"],
        summary: options?.reasoningSummary || "auto",
      };
      params.include!.push("reasoning.encrypted_content");
    } else if (model.thinkingLevelMap?.off !== null) {
      params.reasoning = {
        effort: (model.thinkingLevelMap?.off ??
          "none") as NonNullable<ResponseCreateParamsStreaming["reasoning"]>["effort"],
      };
    }
  }
  return params;
}

interface TextSignatureV1 {
  v: 1;
  id: string;
  phase?: "commentary" | "final_answer";
}

function encodeTextSignature(id: string, phase?: string): string {
  const payload: TextSignatureV1 = { v: 1, id };
  if (phase === "commentary" || phase === "final_answer") {
    payload.phase = phase;
  }
  return JSON.stringify(payload);
}

function parseTextSignature(signature: string | undefined): TextSignatureV1 | undefined {
  if (!signature) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(signature);
    if (parsed && parsed.v === 1 && typeof parsed.id === "string") {
      return parsed as TextSignatureV1;
    }
  } catch {
    // Legacy plain-string signature.
  }
  return { v: 1, id: signature };
}

function citationToAnnotation(citation: UrlCitation): ResponseOutputText.URLCitation {
  return {
    type: "url_citation",
    url: citation.url,
    title: citation.title,
    start_index: citation.startIndex,
    end_index: citation.endIndex,
  };
}

function annotationToCitation(annotation: ResponseOutputText.URLCitation): UrlCitation {
  return {
    type: "url_citation",
    url: annotation.url,
    title: annotation.title,
    startIndex: annotation.start_index,
    endIndex: annotation.end_index,
  };
}

export function convertNativeMessages(
  model: Model<typeof OPENAI_NATIVE_API>,
  context: Context,
): ResponseInput {
  const items: ResponseInputItem[] = [];
  if (context.systemPrompt) {
    items.push({
      role: model.reasoning ? "developer" : "system",
      content: context.systemPrompt,
    });
  }
  let msgIndex = 0;
  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        items.push({ role: "user", content: [{ type: "input_text", text: msg.content }] });
      } else {
        const content = msg.content.map((item) =>
          item.type === "text"
            ? ({ type: "input_text", text: item.text } as const)
            : ({
                type: "input_image",
                detail: "auto",
                image_url: `data:${item.mimeType};base64,${item.data}`,
              } as const),
        );
        if (content.length > 0) {
          items.push({ role: "user", content });
        }
      }
    } else if (msg.role === "assistant") {
      const native = msg.api === OPENAI_NATIVE_API && msg.provider === model.provider;
      for (const block of msg.content as HostedContentBlock[]) {
        if (block.type === "thinking") {
          if (native && block.thinkingSignature) {
            items.push(JSON.parse(block.thinkingSignature) as ResponseReasoningItem);
          }
          // Foreign thinking blocks carry signatures this API cannot verify; drop them.
        } else if (block.type === "text") {
          const signature = native ? parseTextSignature(block.textSignature) : undefined;
          items.push({
            type: "message",
            role: "assistant",
            status: "completed",
            id: signature?.id ?? `msg_${msgIndex}`,
            content: [
              {
                type: "output_text",
                text: block.text,
                annotations: getCitations(block).map(citationToAnnotation),
              },
            ],
            ...(signature?.phase ? { phase: signature.phase } : {}),
          } as ResponseInputItem);
        } else if (block.type === "toolCall") {
          const [callId, itemId] = block.id.split("|");
          items.push({
            type: "function_call",
            // Replaying item ids from another model trips OpenAI's
            // reasoning/function-call pairing validation; omit them.
            id: native && msg.model === model.id ? itemId : undefined,
            call_id: callId ?? block.id,
            name: block.name,
            arguments: JSON.stringify(block.arguments),
          });
        } else if (block.type === "hostedToolCall") {
          if (native) {
            items.push(block.raw);
          }
          // Foreign hosted-tool blocks cannot be replayed to this API.
        }
      }
    } else if (msg.role === "toolResult") {
      const textResult = msg.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");
      const images = msg.content.filter((c) => c.type === "image");
      const [callId] = msg.toolCallId.split("|");
      let outputValue: ResponseInputItem.FunctionCallOutput["output"];
      if (images.length > 0 && model.input.includes("image")) {
        outputValue = [
          ...(textResult.length > 0 ? [{ type: "input_text", text: textResult } as const] : []),
          ...images.map(
            (img) =>
              ({
                type: "input_image",
                detail: "auto",
                image_url: `data:${img.mimeType};base64,${img.data}`,
              }) as const,
          ),
        ];
      } else {
        outputValue = textResult.length > 0 ? textResult : "(see attached image)";
      }
      items.push({
        type: "function_call_output",
        call_id: callId ?? msg.toolCallId,
        output: outputValue,
      });
    }
    msgIndex++;
  }
  return items;
}

// ---------------------------------------------------------------------------
// Responses API stream -> pi-ai events
// ---------------------------------------------------------------------------

async function processNativeStream(
  openaiStream: AsyncIterable<ResponseStreamEvent>,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  model: Model<typeof OPENAI_NATIVE_API>,
): Promise<void> {
  const blocks = output.content as HostedContentBlock[];
  let currentBlock: HostedContentBlock | null = null;
  const blockIndex = () => blocks.length - 1;

  for await (const event of openaiStream) {
    if (event.type === "response.created") {
      output.responseId = event.response.id;
    } else if (event.type === "response.output_item.added") {
      const item = event.item;
      if (item.type === "reasoning") {
        currentBlock = { type: "thinking", thinking: "" };
        blocks.push(currentBlock);
        stream.push({ type: "thinking_start", contentIndex: blockIndex(), partial: output });
      } else if (item.type === "message") {
        currentBlock = { type: "text", text: "", citations: [] } as HostedTextContent;
        blocks.push(currentBlock);
        stream.push({ type: "text_start", contentIndex: blockIndex(), partial: output });
      } else if (item.type === "function_call") {
        currentBlock = {
          type: "toolCall",
          id: `${item.call_id}|${item.id}`,
          name: item.name,
          arguments: {},
          partialJson: item.arguments || "",
        } as ToolCall;
        blocks.push(currentBlock);
        stream.push({ type: "toolcall_start", contentIndex: blockIndex(), partial: output });
      } else if (item.type === "web_search_call") {
        const webSearch = item as ResponseFunctionWebSearch;
        currentBlock = {
          type: "hostedToolCall",
          id: webSearch.id,
          toolName: "web_search",
          status: webSearch.status,
          raw: webSearch,
        };
        blocks.push(currentBlock);
        // No pi-ai event type models hosted tool calls; the block is visible
        // on `partial` in subsequent events and on the final message.
      }
    } else if (event.type === "response.output_text.delta") {
      if (currentBlock?.type === "text") {
        currentBlock.text += event.delta;
        stream.push({
          type: "text_delta",
          contentIndex: blockIndex(),
          delta: event.delta,
          partial: output,
        });
      }
    } else if (event.type === "response.refusal.delta") {
      if (currentBlock?.type === "text") {
        currentBlock.text += event.delta;
        stream.push({
          type: "text_delta",
          contentIndex: blockIndex(),
          delta: event.delta,
          partial: output,
        });
      }
    } else if (event.type === "response.output_text.annotation.added") {
      if (currentBlock?.type === "text") {
        const annotation = event.annotation as { type?: string };
        if (annotation?.type === "url_citation") {
          const textBlock = currentBlock as HostedTextContent;
          textBlock.citations = textBlock.citations ?? [];
          textBlock.citations.push(
            annotationToCitation(annotation as ResponseOutputText.URLCitation),
          );
        }
      }
    } else if (
      event.type === "response.reasoning_summary_text.delta" ||
      event.type === "response.reasoning_text.delta"
    ) {
      if (currentBlock?.type === "thinking") {
        currentBlock.thinking += event.delta;
        stream.push({
          type: "thinking_delta",
          contentIndex: blockIndex(),
          delta: event.delta,
          partial: output,
        });
      }
    } else if (event.type === "response.reasoning_summary_part.done") {
      if (currentBlock?.type === "thinking") {
        currentBlock.thinking += "\n\n";
        stream.push({
          type: "thinking_delta",
          contentIndex: blockIndex(),
          delta: "\n\n",
          partial: output,
        });
      }
    } else if (event.type === "response.function_call_arguments.delta") {
      if (currentBlock?.type === "toolCall") {
        const toolBlock = currentBlock as ToolCall & { partialJson?: string };
        toolBlock.partialJson = (toolBlock.partialJson ?? "") + event.delta;
        toolBlock.arguments = parseStreamingJson(toolBlock.partialJson);
        stream.push({
          type: "toolcall_delta",
          contentIndex: blockIndex(),
          delta: event.delta,
          partial: output,
        });
      }
    } else if (event.type === "response.output_item.done") {
      const item = event.item;
      if (item.type === "reasoning" && currentBlock?.type === "thinking") {
        const summaryText = item.summary?.map((s) => s.text).join("\n\n") || "";
        const contentText = item.content?.map((c) => c.text).join("\n\n") || "";
        currentBlock.thinking = summaryText || contentText || currentBlock.thinking;
        currentBlock.thinkingSignature = JSON.stringify(item);
        stream.push({
          type: "thinking_end",
          contentIndex: blockIndex(),
          content: currentBlock.thinking,
          partial: output,
        });
        currentBlock = null;
      } else if (item.type === "message" && currentBlock?.type === "text") {
        const textBlock = currentBlock as HostedTextContent;
        textBlock.text = item.content
          .map((c) => (c.type === "output_text" ? c.text : c.type === "refusal" ? c.refusal : ""))
          .join("");
        textBlock.textSignature = encodeTextSignature(
          item.id,
          (item as { phase?: string }).phase,
        );
        // The completed item carries the authoritative annotation list.
        const finalCitations = item.content.flatMap((c) =>
          c.type === "output_text"
            ? c.annotations
                .filter((a): a is ResponseOutputText.URLCitation => a.type === "url_citation")
                .map(annotationToCitation)
            : [],
        );
        if (finalCitations.length > 0 || (textBlock.citations?.length ?? 0) > 0) {
          textBlock.citations = finalCitations;
        }
        stream.push({
          type: "text_end",
          contentIndex: blockIndex(),
          content: textBlock.text,
          partial: output,
        });
        currentBlock = null;
      } else if (item.type === "function_call" && currentBlock?.type === "toolCall") {
        const toolBlock = currentBlock as ToolCall & { partialJson?: string };
        toolBlock.arguments = parseStreamingJson(toolBlock.partialJson || item.arguments || "{}");
        delete toolBlock.partialJson;
        stream.push({
          type: "toolcall_end",
          contentIndex: blockIndex(),
          toolCall: toolBlock,
          partial: output,
        });
        currentBlock = null;
      } else if (item.type === "web_search_call") {
        const webSearch = item as ResponseFunctionWebSearch;
        const block = blocks.find(
          (b): b is HostedToolCallContent => isHostedToolCall(b) && b.id === webSearch.id,
        );
        if (block) {
          block.status = webSearch.status;
          block.raw = webSearch;
        } else {
          blocks.push({
            type: "hostedToolCall",
            id: webSearch.id,
            toolName: "web_search",
            status: webSearch.status,
            raw: webSearch,
          });
        }
        currentBlock = null;
      }
    } else if (event.type === "response.completed") {
      const response = event.response;
      if (response?.id) {
        output.responseId = response.id;
      }
      if (response?.usage) {
        const cachedTokens = response.usage.input_tokens_details?.cached_tokens || 0;
        output.usage = {
          input: (response.usage.input_tokens || 0) - cachedTokens,
          output: response.usage.output_tokens || 0,
          cacheRead: cachedTokens,
          cacheWrite: 0,
          totalTokens: response.usage.total_tokens || 0,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
        };
      }
      calculateCost(model, output.usage);
      output.stopReason = mapStopReason(response?.status);
      if (
        output.stopReason === "stop" &&
        output.content.some((b) => b.type === "toolCall")
      ) {
        output.stopReason = "toolUse";
      }
    } else if (event.type === "error") {
      throw new Error(`Error Code ${event.code}: ${event.message}`);
    } else if (event.type === "response.failed") {
      const error = event.response?.error;
      const details = event.response?.incomplete_details;
      const msg = error
        ? `${error.code || "unknown"}: ${error.message || "no message"}`
        : details?.reason
          ? `incomplete: ${details.reason}`
          : "Unknown error (no error details in response)";
      throw new Error(msg);
    }
  }
}

function mapStopReason(status: string | undefined): StopReason {
  switch (status) {
    case undefined:
    case "completed":
    case "in_progress":
    case "queued":
      return "stop";
    case "incomplete":
      return "length";
    case "failed":
    case "cancelled":
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
