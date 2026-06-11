/**
 * "openai-native": a pi-ai API provider for the OpenAI Responses API with
 * native (server-side, "hosted") tools: web search, file search, code
 * interpreter, and image generation.
 *
 * pi-ai's stock "openai-responses" provider discards `url_citation`
 * annotations (it hardcodes `annotations: []` on replay) and only models
 * client-side function tools. This provider is a conversion layer built on
 * the official `openai` SDK that:
 *
 *  - sends enabled hosted tools (web search on by default; the others are
 *    opt-in via options) alongside any client function tools;
 *  - captures citation annotations as structured `citations` on text blocks
 *    (extra fields survive structural typing and JSON serialization);
 *  - captures raw hosted tool call output items as `hostedToolCall` blocks
 *    and replays them verbatim on later turns.
 *
 * Only rath's own code switches over the extended block types. Contexts that
 * contain extended blocks must be flattened before being sent to a stock
 * provider (not implemented here; see issue #5 "out of scope").
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
  FileSearchTool,
  Tool as OpenAITool,
  ResponseCodeInterpreterToolCall,
  ResponseCreateParamsStreaming,
  ResponseFileSearchToolCall,
  ResponseFunctionWebSearch,
  ResponseIncludable,
  ResponseInput,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputText,
  ResponseReasoningItem,
  ResponseStreamEvent,
  ResponseUsage,
  WebSearchTool,
} from "openai/resources/responses/responses.js";
import {
  type Citation,
  getCitations,
  type HostedContentBlock,
  type HostedTextContent,
  type HostedToolCallContent,
  isHostedToolCall,
} from "../hosted-tools.js";

export const OPENAI_NATIVE_API = "openai-native";

/**
 * Hosted (server-side) tools this provider supports. The full enumeration of
 * hosted tools lives in OpenAI's built-in tools documentation
 * (https://platform.openai.com/docs/guides/tools) and in the `Tool` union of
 * the openai SDK (openai/resources/responses/responses). Our support is not
 * meant to be comprehensive — e.g. remote MCP (`mcp_call`, `mcp_list_tools`,
 * `mcp_approval_request`) is not supported.
 */
export type HostedToolName = "web_search" | "file_search" | "code_interpreter" | "image_generation";

/** Raw Responses API output items for the supported hosted tool calls. */
export type HostedToolCallItem =
  | ResponseFunctionWebSearch
  | ResponseFileSearchToolCall
  | ResponseCodeInterpreterToolCall
  | ResponseOutputItem.ImageGenerationCall;

/** A hosted tool call block as produced by this provider. */
export type OpenAINativeHostedToolCall = HostedToolCallContent<HostedToolCallItem> & {
  toolName: HostedToolName;
  status: HostedToolCallItem["status"];
};

export interface OpenAINativeOptions extends StreamOptions {
  reasoningEffort?: "minimal" | "low" | "medium" | "high" | "xhigh";
  reasoningSummary?: "auto" | "detailed" | "concise" | null;
  /**
   * Hosted web search tool configuration. `true` (default) sends
   * `{type: "web_search"}`; `false` disables it; an object customizes
   * filters/user_location.
   */
  webSearch?: boolean | Omit<WebSearchTool, "type">;
  /**
   * Hosted file search over vector stores. Off unless configured — it
   * requires vector store ids.
   */
  fileSearch?: Omit<FileSearchTool, "type">;
  /**
   * Hosted code interpreter. Off by default; `true` uses an auto container,
   * an object customizes the container/files.
   */
  codeInterpreter?: boolean | Omit<OpenAITool.CodeInterpreter, "type">;
  /** Hosted image generation. Off by default; an object customizes output. */
  imageGeneration?: boolean | Omit<OpenAITool.ImageGeneration, "type">;
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
    streamSimple: (model, context, options) => {
      // Clamp to the model's supported levels so an unsupported request (e.g.
      // "xhigh" on a model without it) does not 400 every turn.
      const clamped = options?.reasoning ? clampThinkingLevel(model, options.reasoning) : undefined;
      return streamOpenAINative(model, context, {
        ...options,
        reasoningEffort: clamped === "off" ? undefined : clamped,
      });
    },
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
        throw new Error("OpenAI API key is required. Set OPENAI_API_KEY or pass options.apiKey.");
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
  const include: ResponseIncludable[] = [];
  if (options?.webSearch !== false) {
    include.push("web_search_call.action.sources");
  }
  if (options?.fileSearch) {
    include.push("file_search_call.results");
  }
  if (options?.codeInterpreter) {
    include.push("code_interpreter_call.outputs");
  }
  const params: ResponseCreateParamsStreaming = {
    model: model.id,
    input: convertNativeMessages(model, context),
    stream: true,
    store: false,
    include,
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
  if (options?.fileSearch) {
    tools.push({ type: "file_search", ...options.fileSearch });
  }
  if (options?.codeInterpreter) {
    const config =
      typeof options.codeInterpreter === "object"
        ? options.codeInterpreter
        : { container: { type: "auto" as const } };
    tools.push({ type: "code_interpreter", ...config });
  }
  if (options?.imageGeneration) {
    const config = typeof options.imageGeneration === "object" ? options.imageGeneration : {};
    tools.push({ type: "image_generation", ...config });
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
        effort: (model.thinkingLevelMap?.off ?? "none") as NonNullable<
          ResponseCreateParamsStreaming["reasoning"]
        >["effort"],
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

type OutputAnnotation =
  | ResponseOutputText.URLCitation
  | ResponseOutputText.FileCitation
  | ResponseOutputText.ContainerFileCitation;

function citationToAnnotation(citation: Citation): OutputAnnotation {
  switch (citation.type) {
    case "url_citation":
      return {
        type: "url_citation",
        url: citation.url,
        title: citation.title,
        start_index: citation.startIndex,
        end_index: citation.endIndex,
      };
    case "file_citation":
      return {
        type: "file_citation",
        file_id: citation.fileId,
        filename: citation.filename,
        index: citation.index,
      };
    case "container_file_citation":
      return {
        type: "container_file_citation",
        container_id: citation.containerId,
        file_id: citation.fileId,
        filename: citation.filename,
        start_index: citation.startIndex,
        end_index: citation.endIndex,
      };
  }
}

/** Returns undefined for annotation types we do not capture (e.g. file_path). */
function annotationToCitation(annotation: { type: string }): Citation | undefined {
  switch (annotation.type) {
    case "url_citation": {
      const a = annotation as ResponseOutputText.URLCitation;
      return {
        type: "url_citation",
        url: a.url,
        title: a.title,
        startIndex: a.start_index,
        endIndex: a.end_index,
      };
    }
    case "file_citation": {
      const a = annotation as ResponseOutputText.FileCitation;
      return {
        type: "file_citation",
        fileId: a.file_id,
        filename: a.filename,
        index: a.index,
      };
    }
    case "container_file_citation": {
      const a = annotation as ResponseOutputText.ContainerFileCitation;
      return {
        type: "container_file_citation",
        containerId: a.container_id,
        fileId: a.file_id,
        filename: a.filename,
        startIndex: a.start_index,
        endIndex: a.end_index,
      };
    }
    default:
      return undefined;
  }
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
      // Reasoning items and item ids carry encrypted, model-specific state;
      // replaying them to a different model fails the API's pairing/decryption
      // checks. Only replay them when the producing model matches.
      const sameModel = native && msg.model === model.id;
      let blockIndex = 0;
      for (const block of msg.content as HostedContentBlock[]) {
        if (block.type === "thinking") {
          if (sameModel && block.thinkingSignature) {
            // A hand-edited or corrupted signature must not crash the turn.
            try {
              items.push(JSON.parse(block.thinkingSignature) as ResponseReasoningItem);
            } catch {
              // Skip an unparseable reasoning signature.
            }
          }
          // Foreign or cross-model thinking blocks cannot be replayed here.
        } else if (block.type === "text") {
          // Empty text blocks (e.g. from an errored turn) are rejected by the
          // API as malformed message items.
          if (block.text.length === 0) {
            continue;
          }
          const signature = native ? parseTextSignature(block.textSignature) : undefined;
          // Ids are per text item; key fabricated ids by block, not message,
          // so multi-text messages do not collide. Real ids belong to the
          // producing model — fabricate one on cross-model replay.
          const fabricatedId = `msg_${msgIndex}_${blockIndex}`;
          items.push({
            type: "message",
            role: "assistant",
            status: "completed",
            id: sameModel ? (signature?.id ?? fabricatedId) : fabricatedId,
            content: [
              {
                type: "output_text",
                text: block.text,
                // Citations reference the hosted search call, which is dropped
                // on cross-model replay; drop the annotations with it.
                annotations: sameModel ? getCitations(block).map(citationToAnnotation) : [],
              },
            ],
            ...(sameModel && signature?.phase ? { phase: signature.phase } : {}),
          } as ResponseInputItem);
        } else if (block.type === "toolCall") {
          const [callId, itemId] = block.id.split("|");
          items.push({
            type: "function_call",
            // Replaying item ids from another model trips OpenAI's
            // reasoning/function-call pairing validation; omit them, and the
            // literal "undefined" a missing item id would stringify to.
            id: sameModel && itemId && itemId !== "undefined" ? itemId : undefined,
            call_id: callId ?? block.id,
            name: block.name,
            arguments: JSON.stringify(block.arguments),
          });
        } else if (block.type === "hostedToolCall") {
          if (sameModel) {
            // Blocks on native messages were produced by this provider; their
            // raw items only replay cleanly to the same model.
            items.push(block.raw as HostedToolCallItem);
          }
          // Foreign or cross-model hosted-tool blocks cannot be replayed here.
        }
        blockIndex++;
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

const HOSTED_CALL_ITEMS: Record<HostedToolCallItem["type"], HostedToolName> = {
  web_search_call: "web_search",
  file_search_call: "file_search",
  code_interpreter_call: "code_interpreter",
  image_generation_call: "image_generation",
};

function asHostedToolCallItem(item: { type: string }): HostedToolCallItem | undefined {
  return item.type in HOSTED_CALL_ITEMS ? (item as HostedToolCallItem) : undefined;
}

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
      } else {
        const hosted = asHostedToolCallItem(item);
        if (hosted) {
          currentBlock = {
            type: "hostedToolCall",
            id: hosted.id,
            toolName: HOSTED_CALL_ITEMS[hosted.type],
            status: hosted.status,
            raw: hosted,
          };
          blocks.push(currentBlock);
          // No pi-ai event type models hosted tool calls; the block is
          // visible on `partial` in subsequent events and on the final
          // message.
        }
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
        const citation = annotationToCitation(event.annotation as { type: string });
        if (citation) {
          const textBlock = currentBlock as HostedTextContent;
          textBlock.citations = textBlock.citations ?? [];
          textBlock.citations.push(citation);
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
        textBlock.textSignature = encodeTextSignature(item.id, (item as { phase?: string }).phase);
        // The completed item carries the authoritative annotation list.
        const finalCitations = item.content.flatMap((c) =>
          c.type === "output_text"
            ? c.annotations.flatMap((a) => annotationToCitation(a) ?? [])
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
      } else {
        const hosted = asHostedToolCallItem(item);
        if (hosted) {
          const block = blocks.find(
            (b): b is HostedToolCallContent => isHostedToolCall(b) && b.id === hosted.id,
          );
          if (block) {
            block.status = hosted.status;
            block.raw = hosted;
          } else {
            blocks.push({
              type: "hostedToolCall",
              id: hosted.id,
              toolName: HOSTED_CALL_ITEMS[hosted.type],
              status: hosted.status,
              raw: hosted,
            });
          }
          currentBlock = null;
        }
      }
    } else if (event.type === "response.completed") {
      finalizeUsage(event.response, output, model);
      // A function call that received argument deltas but no output_item.done
      // (rare, but the SDK does not guarantee one) would otherwise persist its
      // scratch buffer; finalize and strip it here too.
      for (const block of output.content) {
        if (block.type === "toolCall" && "partialJson" in block) {
          const tool = block as ToolCall & { partialJson?: string };
          tool.arguments = parseStreamingJson(tool.partialJson || "{}");
          delete tool.partialJson;
        }
      }
      output.stopReason = mapStopReason(event.response?.status);
      if (output.stopReason === "stop" && output.content.some((b) => b.type === "toolCall")) {
        output.stopReason = "toolUse";
      }
    } else if (event.type === "response.incomplete") {
      // The response was cut off (typically max_output_tokens). Drop any tool
      // call still carrying the streaming scratch buffer, and any hosted tool
      // call that never completed — replaying either (truncated arguments, or
      // an in-progress hosted item) is rejected by the API and wedges the
      // session. Report length rather than a clean stop.
      finalizeUsage(event.response, output, model);
      output.content = (output.content as HostedContentBlock[]).filter(
        (b) =>
          !(b.type === "toolCall" && "partialJson" in b) &&
          !(isHostedToolCall(b) && b.status !== "completed"),
      ) as AssistantMessage["content"];
      output.stopReason = "length";
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

/** Capture token usage and cost from a terminal response object. */
function finalizeUsage(
  response: { id?: string; usage?: ResponseUsage } | undefined,
  output: AssistantMessage,
  model: Model<typeof OPENAI_NATIVE_API>,
): void {
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
