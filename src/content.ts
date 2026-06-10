/**
 * Extended content blocks shared by rath providers, and utilities for
 * working with them. Provider-agnostic: openai-native produces these today;
 * an Anthropic provider with server-side tools will produce the same shapes
 * with its own raw items.
 *
 * The conventions:
 * - Citations ride along as a `citations` array on text blocks.
 * - Server-side ("hosted") tool calls ride along as `hostedToolCall` blocks
 *   whose `raw` is the provider's verbatim wire item, replayed by the
 *   provider that produced it.
 * - Extra fields and blocks survive pi-ai's structural typing and JSON
 *   serialization; only rath code switches over them. Code that consumes
 *   assistant messages must tolerate unknown block types.
 */
import type {
  AssistantMessage,
  TextContent,
  ThinkingContent,
  ToolCall,
} from "@earendil-works/pi-ai";

/** Structured citation extracted from a `url_citation` annotation (web search). */
export interface UrlCitation {
  type: "url_citation";
  url: string;
  title: string;
  /** Index of the first character of the cited span in the text block. */
  startIndex: number;
  /** Index one past the last character of the cited span in the text block. */
  endIndex: number;
}

/** Structured citation extracted from a `file_citation` annotation (file search). */
export interface FileCitation {
  type: "file_citation";
  fileId: string;
  filename: string;
  /** Index of the file in the list of files. */
  index: number;
}

/**
 * Structured citation extracted from a `container_file_citation` annotation
 * (a file produced by the code interpreter).
 */
export interface ContainerFileCitation {
  type: "container_file_citation";
  containerId: string;
  fileId: string;
  filename: string;
  /** Index of the first character of the cited span in the text block. */
  startIndex: number;
  /** Index one past the last character of the cited span in the text block. */
  endIndex: number;
}

export type Citation = UrlCitation | FileCitation | ContainerFileCitation;

/** Text block extended with citations. Structurally still a TextContent. */
export interface HostedTextContent extends TextContent {
  citations?: Citation[];
}

/**
 * A server-side tool call executed by the provider. `raw` is the provider's
 * verbatim wire item; it is replayed to that provider on later turns. This
 * block type is not part of pi-ai's content union: only rath code
 * understands it. Providers narrow `raw` (and possibly `toolName`) to their
 * own types.
 */
export interface HostedToolCallContent<TRaw = unknown> {
  type: "hostedToolCall";
  id: string;
  toolName: string;
  status: string;
  raw: TRaw;
}

export type HostedContentBlock = TextContent | ThinkingContent | ToolCall | HostedToolCallContent;

export function isHostedToolCall(block: { type: string }): block is HostedToolCallContent {
  return block.type === "hostedToolCall";
}

/**
 * View an assistant message's content as the extended block union. pi-ai
 * types content as (TextContent | ThinkingContent | ToolCall)[]; messages
 * produced by rath providers additionally carry HostedToolCallContent
 * blocks.
 */
export function contentBlocks(message: AssistantMessage): HostedContentBlock[] {
  return message.content as unknown as HostedContentBlock[];
}

/** Hosted (server-side) tool call blocks on an assistant message. */
export function getHostedToolCalls<TRaw = unknown>(
  message: AssistantMessage,
): HostedToolCallContent<TRaw>[] {
  return contentBlocks(message).filter(isHostedToolCall) as HostedToolCallContent<TRaw>[];
}

/** Citations on a text block, if any. */
export function getCitations(block: TextContent): Citation[] {
  return (block as HostedTextContent).citations ?? [];
}

/**
 * Citation rendering. Citations can be rendered into a "Sources:" text block
 * appended to the assistant message, marked with `renderedCitations: true`.
 * The trailer persists in saved contexts and flattens for free when a
 * context is handed to a provider that does not understand citations.
 * Before replay to the provider that produced the citations — which
 * reconstructs the real annotations itself — strip the marked blocks with
 * stripRenderedCitations, so the model's own history stays byte-identical
 * to what it produced.
 */
export interface RenderedCitationsBlock {
  type: "text";
  text: string;
  renderedCitations: true;
}

export function isRenderedCitations(block: { type: string }): block is RenderedCitationsBlock {
  return block.type === "text" && (block as RenderedCitationsBlock).renderedCitations === true;
}

/** URL citations across the message's text blocks, deduplicated by URL. */
export function uniqueUrlCitations(message: AssistantMessage): UrlCitation[] {
  const seen = new Map<string, UrlCitation>();
  for (const block of message.content) {
    if (block.type === "text" && !isRenderedCitations(block)) {
      for (const citation of getCitations(block)) {
        if (citation.type === "url_citation" && !seen.has(citation.url)) {
          seen.set(citation.url, citation);
        }
      }
    }
  }
  return [...seen.values()];
}

/**
 * Render the message's citations and append them as a marked text block.
 * Idempotent. Returns the trailer text, or undefined when there is nothing
 * to add.
 */
export function applyCitationTrailer(message: AssistantMessage): string | undefined {
  if (message.content.some(isRenderedCitations)) {
    return undefined;
  }
  const citations = uniqueUrlCitations(message);
  if (citations.length === 0) {
    return undefined;
  }
  const lines = citations.map((c) => `- ${c.title ? `${c.title} — ` : ""}${c.url}`);
  const trailer = `Sources:\n${lines.join("\n")}`;
  const block: RenderedCitationsBlock = { type: "text", text: trailer, renderedCitations: true };
  (message.content as unknown as RenderedCitationsBlock[]).push(block);
  return trailer;
}

/**
 * Copy of the message without rendered-citations trailer blocks. Use before
 * replaying history to the provider that produced the citations, and for
 * displays that render citations their own way.
 */
export function stripRenderedCitations(message: AssistantMessage): AssistantMessage {
  if (!message.content.some(isRenderedCitations)) {
    return message;
  }
  return { ...message, content: message.content.filter((b) => !isRenderedCitations(b)) };
}
