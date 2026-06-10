/**
 * `rath run`: a generic agent loop with nothing implicit.
 *
 * No skill discovery, no context-file walking (AGENTS.md is never read), no
 * tools unless explicitly enabled via --tools. The model sees exactly what
 * the flags specify: the system prompt, the loaded context, and the user's
 * prompts. The only thing taken from the environment is the provider API
 * key. rath development itself is meant to happen inside `rath run`.
 *
 * Built on pi-agent-core's Agent (the stateful loop wrapper: transcript,
 * lifecycle events, tool execution, model switching via state.model) and
 * pi-ai's provider registry; the openai-native provider is registered, so
 * hosted web search is available (on by default, --no-web-search disables
 * it). Client-side tools are loaded on demand from
 * @earendil-works/pi-coding-agent. Frontends: a plain readline REPL
 * (default, also used for -p one-shots) and a pi-tui interface (-T/--tui).
 */
import { readFileSync, writeFileSync } from "node:fs";
import * as readline from "node:readline/promises";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
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
  type SimpleStreamOptions,
  streamSimple,
  type TextContent,
} from "@earendil-works/pi-ai";
import { type Command, fullName, helpRequested, helpText } from "../command.js";
import {
  contentBlocks,
  getCitations,
  isHostedToolCall,
  OPENAI_NATIVE_API,
  openaiNativeModel,
  registerOpenAINative,
} from "../index.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function dim(text: string): string {
  return process.stdout.isTTY ? `${DIM}${text}${RESET}` : text;
}

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
 * Imported lazily: the package is large and only needed when --tools is used.
 */
export async function loadTools(names: ToolName[], cwd: string): Promise<AgentTool[]> {
  const pi = await import("@earendil-works/pi-coding-agent");
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

interface Renderer {
  /** True when the most recent run produced an error message. */
  hadError: () => boolean;
}

/** Wire plain stdout rendering onto the agent's event stream. */
function attachRenderer(agent: Agent): Renderer {
  let openLine = false;
  const ensureNewline = () => {
    if (openLine) {
      process.stdout.write("\n");
      openLine = false;
    }
  };
  const seenHostedCalls = new Set<string>();
  let errored = false;

  agent.subscribe((event) => {
    if (event.type === "agent_start") {
      errored = false;
    } else if (event.type === "message_update") {
      const e = event.assistantMessageEvent;
      if (e.type === "text_delta") {
        process.stdout.write(e.delta);
        openLine = true;
      } else if (e.type === "thinking_delta") {
        process.stdout.write(dim(e.delta));
        openLine = true;
      } else if (e.type === "text_end" || e.type === "thinking_end") {
        ensureNewline();
      }
      // Hosted tool calls emit no assistant-message events; they appear as
      // blocks on the partial message. Announce each one once.
      if (event.message.role === "assistant") {
        for (const block of contentBlocks(event.message)) {
          if (isHostedToolCall(block) && !seenHostedCalls.has(block.id)) {
            seenHostedCalls.add(block.id);
            ensureNewline();
            process.stdout.write(`${dim(`[${block.toolName}]`)}\n`);
          }
        }
      }
    } else if (event.type === "tool_execution_start") {
      ensureNewline();
      const args = JSON.stringify(event.args);
      process.stdout.write(
        `${dim(`[${event.toolName}: ${args.length > 120 ? `${args.slice(0, 120)}…` : args}]`)}\n`,
      );
    } else if (event.type === "message_end") {
      const message = event.message;
      if (message.role === "assistant") {
        ensureNewline();
        if (message.stopReason === "error" || message.stopReason === "aborted") {
          process.stderr.write(`error: ${message.errorMessage ?? "unknown"}\n`);
          errored = true;
          return;
        }
        const trailer = applyCitationTrailer(message);
        if (trailer) {
          process.stdout.write(`${dim(trailer)}\n`);
        }
      }
    }
  });
  return { hadError: () => errored };
}

export const runCommand: Command = {
  name: "run",
  summary: "Run a generic agent loop",
  description:
    "Starts an agent loop with nothing implicit: no skill discovery, no\n" +
    "context-file walking, no tools unless explicitly enabled. The model sees\n" +
    "exactly what the flags specify. The provider API key (e.g.\n" +
    "OPENAI_API_KEY) is the only input taken from the environment.\n" +
    "\n" +
    "Without --prompt, reads prompts interactively (plain REPL, or pi-tui\n" +
    "with -T/--tui). In-session commands: /info shows the configuration,\n" +
    "/model [provider/model-id] shows or\n" +
    "switches the model, /lsmodels [filter] lists available models,\n" +
    "/reasoning [level] shows or sets the reasoning level, /exit quits.\n" +
    "With --prompt, runs one prompt and exits (0 on success).",
  flags: [
    {
      long: "model",
      short: "m",
      takesValue: true,
      description: "Model as <provider>/<model-id> (default: openai-native/gpt-5-mini)",
    },
    {
      long: "prompt",
      short: "p",
      takesValue: true,
      description: "Run a single prompt non-interactively and exit",
    },
    {
      long: "tui",
      short: "T",
      takesValue: false,
      description: "Interactive pi-tui interface instead of the plain REPL",
    },
    {
      long: "system-prompt",
      takesValue: true,
      description: "System prompt text (default: a one-line generic assistant prompt)",
    },
    {
      long: "system-prompt-file",
      takesValue: true,
      description: "Read the system prompt from a file",
    },
    {
      long: "reasoning",
      takesValue: true,
      description: "Reasoning effort: off, minimal, low, medium, high, xhigh (default: low)",
    },
    {
      long: "no-web-search",
      takesValue: false,
      description: "Disable hosted web search (openai-native)",
    },
    {
      long: "tools",
      takesValue: true,
      repeatable: true,
      description:
        "Enable client-side tools (comma-separated or repeated): read, bash, edit, write, " +
        "grep, find, ls. They run with your privileges in the current directory.",
    },
    {
      long: "load",
      takesValue: true,
      description: "Load a previously saved context (JSON) before starting",
    },
    {
      long: "save",
      takesValue: true,
      description: "Save the context (JSON) to this path on exit",
    },
  ],
  async run(prefix, argv) {
    if (helpRequested(argv)) {
      process.stdout.write(`${helpText(this, prefix)}\n`);
      return 0;
    }

    const flags: RunFlags = {
      model: `${OPENAI_NATIVE_API}/gpt-5-mini`,
      systemPrompt: "You are a helpful assistant.",
      reasoning: "low",
      webSearch: true,
      tools: [],
      tui: false,
    };
    let systemPromptFile: string | undefined;
    for (let i = 0; i < argv.length; i++) {
      const token = argv[i]!;
      const value = (): string => {
        const v = argv[++i];
        if (v === undefined) {
          throw new Error(`Option ${token} requires a value`);
        }
        return v;
      };
      try {
        if (token === "-m" || token === "--model") {
          flags.model = value();
        } else if (token === "-p" || token === "--prompt") {
          flags.prompt = value();
        } else if (token === "-T" || token === "--tui") {
          flags.tui = true;
        } else if (token === "--system-prompt") {
          flags.systemPrompt = value();
        } else if (token === "--system-prompt-file") {
          systemPromptFile = value();
        } else if (token === "--reasoning") {
          const level = value() as ReasoningLevel;
          if (!REASONING_LEVELS.includes(level)) {
            process.stderr.write(
              `Invalid --reasoning: ${level} (use ${REASONING_LEVELS.join(", ")})\n`,
            );
            return 1;
          }
          flags.reasoning = level;
        } else if (token === "--no-web-search") {
          flags.webSearch = false;
        } else if (token === "--tools") {
          for (const name of value().split(",")) {
            const trimmed = name.trim() as ToolName;
            if (!TOOL_NAMES.includes(trimmed)) {
              process.stderr.write(`Unknown tool: ${trimmed} (use ${TOOL_NAMES.join(", ")})\n`);
              return 1;
            }
            if (!flags.tools.includes(trimmed)) {
              flags.tools.push(trimmed);
            }
          }
        } else if (token === "--load") {
          flags.loadPath = value();
        } else if (token === "--save") {
          flags.savePath = value();
        } else {
          process.stderr.write(
            `Unknown argument: ${token}\nRun "${fullName(this, prefix)} -h" for usage.\n`,
          );
          return 1;
        }
      } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
        return 1;
      }
    }

    registerOpenAINative();
    let agent: Agent;
    try {
      const model = resolveModel(flags.model);
      if (systemPromptFile) {
        flags.systemPrompt = readFileSync(systemPromptFile, "utf8").trim();
      }
      const loaded = flags.loadPath ? loadContext(flags.loadPath) : undefined;
      agent = new Agent({
        initialState: {
          systemPrompt: loaded?.systemPrompt ?? flags.systemPrompt,
          model,
          thinkingLevel: flags.reasoning,
          messages: loaded?.messages ?? [],
          tools: await loadTools(flags.tools, process.cwd()),
        },
        streamFn: (m, ctx, options) =>
          streamSimple(m, ctx, {
            ...options,
            webSearch: flags.webSearch,
          } as SimpleStreamOptions),
        // openai-native replays real annotations; the rendered trailer would
        // duplicate them. Other providers get the trailer as plain text.
        convertToLlm: (messages) =>
          messages.flatMap((m): Message[] => {
            if (m.role !== "user" && m.role !== "assistant" && m.role !== "toolResult") {
              return [];
            }
            if (m.role === "assistant" && agent.state.model.api === OPENAI_NATIVE_API) {
              return [{ ...m, content: m.content.filter((b) => !isRenderedCitations(b)) }];
            }
            return [m];
          }),
      });
    } catch (error) {
      process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
      return 1;
    }

    const save = () => {
      if (flags.savePath) {
        saveContext(agent, flags.savePath);
        process.stderr.write(`${dim(`context saved to ${flags.savePath}`)}\n`);
      }
    };

    if (flags.prompt !== undefined) {
      const renderer = attachRenderer(agent);
      await agent.prompt(flags.prompt);
      save();
      return renderer.hadError() ? 1 : 0;
    }

    if (flags.tui) {
      const { runTui } = await import("./run-tui.js");
      const code = await runTui(agent, flags);
      save();
      return code;
    }

    attachRenderer(agent);
    process.stderr.write(
      `${dim(`model: ${flags.model} | /info /model /lsmodels /reasoning /exit (or Ctrl+D)`)}\n`,
    );
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    try {
      while (true) {
        let line: string;
        try {
          line = await rl.question("rath> ");
        } catch {
          break; // Ctrl+D / closed input
        }
        const trimmed = line.trim();
        if (trimmed.length === 0) {
          continue;
        }
        const result = handleSlashCommand(trimmed, agent, flags);
        if (result) {
          if (result.output) {
            if (result.isError) {
              process.stderr.write(`${result.output}\n`);
            } else {
              process.stderr.write(`${dim(result.output)}\n`);
            }
          }
          if (result.exit) {
            break;
          }
          continue;
        }
        await agent.prompt(trimmed);
      }
    } finally {
      rl.close();
      save();
    }
    return 0;
  },
};
