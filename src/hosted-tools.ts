/**
 * Hosted (server-side) tool machinery shared by rath providers: the extended
 * content blocks and utilities for working with them. Provider-agnostic:
 * openai-native produces these today; an Anthropic provider will produce
 * the same shapes with its own raw items.
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

/** All citations across the message's text blocks, deduplicated by identity,
 * in order of first appearance. Covers every citation type (web search, file
 * search, code interpreter), unlike uniqueUrlCitations which is URL-only. */
function uniqueCitations(message: AssistantMessage): Citation[] {
  const seen = new Map<string, Citation>();
  for (const block of message.content) {
    if (block.type === "text" && !isRenderedCitations(block)) {
      for (const citation of getCitations(block)) {
        const key =
          citation.type === "url_citation"
            ? `url:${citation.url}`
            : citation.type === "file_citation"
              ? `file:${citation.fileId}`
              : `container:${citation.containerId}:${citation.fileId}`;
        if (!seen.has(key)) {
          seen.set(key, citation);
        }
      }
    }
  }
  return [...seen.values()];
}

function citationLine(c: Citation): string {
  switch (c.type) {
    case "url_citation":
      return `- ${c.title ? `${c.title} — ` : ""}${c.url}`;
    case "file_citation":
      return `- ${c.filename || c.fileId}`;
    case "container_file_citation":
      return `- ${c.filename || c.fileId} (container ${c.containerId})`;
  }
}

/** Render the message's unique citations to "Sources:" trailer text, or
 * undefined when there are none. The single source of truth for trailer text,
 * so the display/persistence trailer and the flattened-handoff trailer match. */
function renderCitationTrailer(message: AssistantMessage): string | undefined {
  const citations = uniqueCitations(message);
  if (citations.length === 0) {
    return undefined;
  }
  return `Sources:\n${citations.map(citationLine).join("\n")}`;
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
  const trailer = renderCitationTrailer(message);
  if (trailer === undefined) {
    return undefined;
  }
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

// ---------------------------------------------------------------------------
// Flatten-on-handoff
// ---------------------------------------------------------------------------

/**
 * Render a hosted tool call's `raw` item to a single descriptive text line.
 *
 * Provider-agnostic: hosted-tools.ts must not depend on a provider SDK, so we
 * probe the raw item structurally for the fields rath providers are known to
 * populate (today: the OpenAI Responses API web_search_call / file_search_call
 * / code_interpreter_call / image_generation_call shapes) rather than import
 * their types. Unknown shapes degrade to a generic "[toolName] (status)" line —
 * the foreign model still learns a server-side tool ran, which is the whole
 * point, even if we cannot name the query.
 */
function renderHostedToolCall(block: HostedToolCallContent): string {
  const raw = block.raw as Record<string, unknown> | null | undefined;
  const label = block.toolName || "hosted tool";
  const status = block.status ? ` (${block.status})` : "";
  const detail = raw ? hostedToolDetail(raw) : "";
  return `[${label}${status}]${detail ? ` ${detail}` : ""}`;
}

/** A flat list of strings from an unknown array field, dropping non-strings. */
function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === "string") : [];
}

/**
 * Pull human-readable detail out of a raw hosted-tool item. Knows the OpenAI
 * Responses API web_search / file_search / code_interpreter shapes
 * structurally; returns "" when it recognizes nothing (e.g. an
 * image_generation_call, which has no text to surface), so the caller still
 * emits the generic label line.
 */
function hostedToolDetail(raw: Record<string, unknown>): string {
  // web_search_call: query/queries + sources live under `action`.
  const action = raw.action as Record<string, unknown> | undefined;
  if (action && typeof action.type === "string") {
    if (action.type === "search") {
      const queries = stringList(action.queries);
      if (queries.length === 0 && typeof action.query === "string") {
        queries.push(action.query);
      }
      const sources = Array.isArray(action.sources)
        ? (action.sources as Record<string, unknown>[])
            .map((s) => (typeof s.url === "string" ? s.url : undefined))
            .filter((u): u is string => u !== undefined)
        : [];
      const parts: string[] = [];
      if (queries.length > 0) {
        parts.push(`searched: ${queries.map((q) => `"${q}"`).join(", ")}`);
      }
      if (sources.length > 0) {
        parts.push(`sources: ${sources.join(", ")}`);
      }
      return parts.join("; ");
    }
    if (action.type === "open_page" && typeof action.url === "string") {
      return `opened: ${action.url}`;
    }
    if (action.type === "find_in_page" && typeof action.url === "string") {
      const pattern = typeof action.pattern === "string" ? ` for "${action.pattern}"` : "";
      return `searched in ${action.url}${pattern}`;
    }
  }
  // file_search_call: top-level `queries`.
  const queries = stringList(raw.queries);
  if (queries.length > 0) {
    return `queried: ${queries.map((q) => `"${q}"`).join(", ")}`;
  }
  // code_interpreter_call: top-level `code`.
  if (typeof raw.code === "string" && raw.code.length > 0) {
    const code = raw.code.replace(/\s+/g, " ").trim();
    return `ran code: ${code.length > 200 ? `${code.slice(0, 200)}…` : code}`;
  }
  return "";
}

/**
 * Flatten an assistant message's extended content into blocks a foreign
 * provider (one whose pi-ai converter only understands text | thinking |
 * toolCall) preserves. Used before handing a context built on one provider to
 * a different provider: that converter SILENTLY DROPS hostedToolCall blocks and
 * ignores inline citations, so the foreign model would otherwise lose all
 * record that a server-side search ran and what it found.
 *
 * The transformation, in original block order:
 *  - `hostedToolCall` blocks -> a descriptive text block ("[web_search]
 *    searched: "…"; sources: …"). PLAIN TEXT, not a synthetic
 *    toolCall+toolResult pair: a toolCall with no matching toolResult (or the
 *    reverse) breaks the provider's call/result pairing and the API rejects the
 *    turn; the hosted call has no client-side counterpart to pair with. Plain
 *    text inserted in place is the simpler, safer representation and reads as
 *    ordinary history.
 *  - `thinking` blocks -> dropped (a foreign provider rejects another
 *    provider's thinking signatures).
 *  - citation-bearing text -> the text is kept as-is and a single rendered
 *    "Sources:" trailer is ensured (the same trailer applyCitationTrailer
 *    produces). If the message already carries that trailer it is reused, so
 *    we never double-render; inline `citations` arrays are left on the text
 *    block (harmless to a foreign provider, which ignores them) and surfaced
 *    through the trailer instead.
 *
 * Idempotent (re-flattening adds no second trailer, and there are no
 * hostedToolCall/thinking blocks left to render) and a no-op (returns the same
 * message reference) for messages with no extended content.
 */
export function flattenHostedContent(message: AssistantMessage): AssistantMessage {
  const blocks = contentBlocks(message);
  const hasHosted = blocks.some(isHostedToolCall);
  const hasThinking = blocks.some((b) => b.type === "thinking");
  const trailer = applyCitationTrailerText(message);
  // No extended content to flatten: same reference back (no-op, idempotent).
  if (!hasHosted && !hasThinking && trailer === undefined) {
    return message;
  }
  const flattened: HostedContentBlock[] = [];
  for (const block of blocks) {
    if (block.type === "thinking") {
      continue;
    }
    // Drop any existing rendered-citations trailer; the single canonical
    // trailer is re-appended below, so keeping this one would double-render on
    // a second flatten pass (idempotency).
    if (isRenderedCitations(block)) {
      continue;
    }
    if (isHostedToolCall(block)) {
      flattened.push({ type: "text", text: renderHostedToolCall(block) });
      continue;
    }
    // Empty text blocks are rejected as malformed message items by both
    // openai-native and foreign providers; drop them (mirrors the converter).
    if (block.type === "text" && block.text.length === 0) {
      continue;
    }
    flattened.push(block);
  }
  if (trailer !== undefined) {
    const block: RenderedCitationsBlock = { type: "text", text: trailer, renderedCitations: true };
    flattened.push(block);
  }
  return { ...message, content: flattened as unknown as AssistantMessage["content"] };
}

/**
 * The citations trailer text this message should carry, without mutating it:
 * the existing trailer if one is present, a freshly rendered one if inline
 * citations exist, or undefined if there is nothing to render. Mirrors
 * applyCitationTrailer's rendering so the flattened trailer matches the
 * display/persistence trailer exactly (no double-render on idempotent reflatten).
 */
function applyCitationTrailerText(message: AssistantMessage): string | undefined {
  const existing = message.content.find(isRenderedCitations);
  if (existing) {
    return (existing as RenderedCitationsBlock).text;
  }
  return renderCitationTrailer(message);
}
