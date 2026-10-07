import {
  type Context,
  getCurrentSystemPrompt,
  getCurrentTools,
  type TranscriptContext,
} from "@earendil-works/pi-ai/compat";

/**
 * Collapse a pi transcript into the prompt/tools/messages shape the native
 * converters were written against. Since pi 1.0 the agent loop carries the
 * system prompt and tool declarations in system messages; a TranscriptContext
 * is still structurally assignable to Context, so without this the native
 * providers would silently send no prompt and no tools.
 */
export function legacyContext(context: TranscriptContext): Context {
  const systemPrompt = getCurrentSystemPrompt(context.messages);
  return {
    ...(systemPrompt.length > 0 && { systemPrompt }),
    tools: getCurrentTools(context.messages),
    messages: context.messages.filter((message) => message.role !== "system"),
  };
}
