/**
 * Shared session logic for `rath run`'s frontends (plain REPL and TUI):
 * flag types, model resolution, tool loading, context persistence, citation
 * trailer merging, and the in-session slash commands.
 */
import { readFileSync, writeFileSync } from "node:fs";
import type { Agent, AgentTool } from "@earendil-works/pi-agent-core";
import {
  type Api,
  type AssistantMessage,
  getModel,
  getModels,
  getProviders,
  getSupportedThinkingLevels,
  type KnownProvider,
  type Message,
  type Model,
  type TextContent,
} from "@earendil-works/pi-ai";
import { getCitations, OPENAI_NATIVE_API, openaiNativeModel } from "../index.js";

export const REASONING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"] as const;
export type ReasoningLevel = (typeof REASONING_LEVELS)[number];

export const TOOL_NAMES = ["read", "bash", "edit", "write", "grep", "find", "ls"] as const;
export type ToolName = (typeof TOOL_NAMES)[number];

export interface RunFlags {
  model: string;
  prompt?: string;
  systemPrompt: string;
  reasoning: ReasoningLevel;
  webSearch: boolean;
  tools: ToolName[];
  tui: boolean;
  loadPath?: string;
  savePath?: string;
}

/**
 * Load the requested client-side tools from @earendil-works/pi-coding-agent.
 * The dependency is imported on demand so that rath's published package does
 * not require it; it is only needed when --tools is used.
 */
export async function loadTools(names: ToolName[], cwd: string): Promise<AgentTool[]> {
  let pi: typeof import("@earendil-works/pi-coding-agent");
  try {
    pi = await import("@earendil-works/pi-coding-agent");
  } catch {
    throw new Error(
      "--tools requires @earendil-works/pi-coding-agent; install it with: " +
        "npm install @earendil-works/pi-coding-agent",
    );
  }
  const factories: Record<ToolName, (cwd: string) => unknown> = {
    read: pi.createReadTool,
    bash: pi.createBashTool,
    edit: pi.createEditTool,
    write: pi.createWriteTool,
    grep: pi.createGrepTool,
    find: pi.createFindTool,
    ls: pi.createLsTool,
  };
  return names.map((name) => factories[name](cwd) as AgentTool);
}

export function resolveModel(spec: string): Model<Api> {
  const slash = spec.indexOf("/");
  if (slash <= 0 || slash === spec.length - 1) {
    throw new Error(`Model must be <provider>/<model-id> (got: ${spec})`);
  }
  const provider = spec.slice(0, slash);
  const modelId = spec.slice(slash + 1);
  if (provider === OPENAI_NATIVE_API) {
    return openaiNativeModel(modelId);
  }
  const model = getModel(provider as KnownProvider, modelId as never) as Model<Api> | undefined;
  if (!model) {
    throw new Error(`Unknown model: ${spec}`);
  }
  return model;
}

/** All selectable model specs, openai-native first. */
export function listModels(filter?: string): string[] {
  const specs: string[] = [];
  for (const model of getModels("openai")) {
    specs.push(`${OPENAI_NATIVE_API}/${model.id}`);
  }
  for (const provider of getProviders()) {
    for (const model of getModels(provider)) {
      specs.push(`${provider}/${model.id}`);
    }
  }
  return filter ? specs.filter((s) => s.includes(filter)) : specs;
}

export interface SerializedContext {
  systemPrompt?: string;
  messages: Message[];
}

export function loadContext(path: string): SerializedContext {
  const parsed = JSON.parse(readFileSync(path, "utf8")) as SerializedContext;
  if (!Array.isArray(parsed.messages)) {
    throw new Error(`${path} does not contain a serialized context (messages array missing)`);
  }
  return parsed;
}

export function saveContext(agent: Agent, path: string): void {
  const messages = agent.state.messages.filter(
    (m): m is Message => m.role === "user" || m.role === "assistant" || m.role === "toolResult",
  );
  writeFileSync(
    path,
    `${JSON.stringify({ systemPrompt: agent.state.systemPrompt, messages }, null, 2)}\n`,
  );
}

/**
 * Citation merging. After each assistant message, citations are rendered
 * into a "Sources:" text block appended to the message, marked with
 * `renderedCitations: true`. The trailer persists in saved contexts and
 * flattens for free when a context is handed to a provider that does not
 * understand citations. Before replay to openai-native — which reconstructs
 * the real annotations itself — marked blocks are stripped, so the model's
 * own history stays byte-identical to what it produced.
 */
export interface RenderedCitationsBlock {
  type: "text";
  text: string;
  renderedCitations: true;
}

export function isRenderedCitations(block: { type: string }): block is RenderedCitationsBlock {
  return block.type === "text" && (block as RenderedCitationsBlock).renderedCitations === true;
}

/**
 * Render the message's citations and append them as a marked text block.
 * Returns the trailer text, or undefined when there is nothing to add.
 */
export function applyCitationTrailer(message: AssistantMessage): string | undefined {
  if (message.content.some(isRenderedCitations)) {
    return undefined;
  }
  const citations = message.content
    .filter((b): b is TextContent => b.type === "text")
    .flatMap(getCitations);
  const urls = new Map<string, string>();
  for (const citation of citations) {
    if (citation.type === "url_citation" && !urls.has(citation.url)) {
      urls.set(citation.url, citation.title);
    }
  }
  if (urls.size === 0) {
    return undefined;
  }
  const lines = [...urls].map(([url, title]) => `- ${title ? `${title} — ` : ""}${url}`);
  const trailer = `Sources:\n${lines.join("\n")}`;
  const block: RenderedCitationsBlock = { type: "text", text: trailer, renderedCitations: true };
  (message.content as unknown as RenderedCitationsBlock[]).push(block);
  return trailer;
}

/** Current session configuration as key/value pairs, for /info displays. */
export function sessionInfo(agent: Agent, flags: RunFlags): [string, string][] {
  const tools = agent.state.tools.map((t) => t.name);
  return [
    ["model", flags.model],
    ["reasoning", String(agent.state.thinkingLevel)],
    ["web search", flags.webSearch ? "on (hosted, openai-native only)" : "off"],
    ["tools", tools.length > 0 ? tools.join(", ") : "none"],
    ["skills", "none (rath run loads no skills or context files)"],
    ["system prompt", "/sys to display"],
    ["messages in context", String(agent.state.messages.length)],
    ...(flags.loadPath ? [["loaded from", flags.loadPath] as [string, string]] : []),
    ["save on exit", flags.savePath ?? "off"],
  ];
}

export interface SlashResult {
  output?: string;
  isError?: boolean;
  exit?: boolean;
}

/**
 * Handle an in-session slash command. Returns undefined when the input is
 * not a slash command (i.e. it is a prompt for the model).
 */
export function handleSlashCommand(
  input: string,
  agent: Agent,
  flags: RunFlags,
): SlashResult | undefined {
  if (!input.startsWith("/")) {
    return undefined;
  }
  const [command = "", ...rest] = input.split(/\s+/);
  const arg = rest.join(" ").trim();
  switch (command) {
    case "/exit":
      return { exit: true };
    case "/info":
      return {
        output: sessionInfo(agent, flags)
          .map(([key, value]) => `${key}: ${value}`)
          .join("\n"),
      };
    case "/sys":
      return { output: agent.state.systemPrompt };
    case "/model": {
      if (arg.length === 0) {
        return { output: `model: ${flags.model}` };
      }
      try {
        agent.state.model = resolveModel(arg);
        flags.model = arg;
        return { output: `model: ${arg}` };
      } catch (error) {
        return { output: error instanceof Error ? error.message : String(error), isError: true };
      }
    }
    case "/lsmodels": {
      const specs = listModels(arg.length > 0 ? arg : undefined);
      const footer = `${specs.length} model${specs.length === 1 ? "" : "s"}${arg ? ` matching "${arg}"` : ""}`;
      return { output: specs.length > 0 ? `${specs.join("\n")}\n${footer}` : footer };
    }
    case "/reasoning": {
      const supported = getSupportedThinkingLevels(agent.state.model as Model<Api>);
      if (arg.length === 0) {
        return {
          output: `reasoning: ${agent.state.thinkingLevel} (supported by ${flags.model}: ${supported.join(", ")})`,
        };
      }
      if (!REASONING_LEVELS.includes(arg as ReasoningLevel)) {
        return {
          output: `Invalid reasoning level: ${arg} (use ${REASONING_LEVELS.join(", ")})`,
          isError: true,
        };
      }
      agent.state.thinkingLevel = arg as ReasoningLevel;
      flags.reasoning = arg as ReasoningLevel;
      const note = supported.includes(arg as ReasoningLevel)
        ? ""
        : ` (outside ${flags.model}'s supported levels — the provider will clamp it)`;
      return { output: `reasoning: ${arg}${note}` };
    }
    default:
      return {
        output: `Unknown command: ${command} (commands: /info, /sys, /model, /lsmodels, /reasoning, /exit)`,
        isError: true,
      };
  }
}
