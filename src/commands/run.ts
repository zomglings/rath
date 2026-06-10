/**
 * `rath run`: a generic agent loop with nothing implicit.
 *
 * No skill discovery, no context-file walking (AGENTS.md is never read), no
 * tools unless explicitly enabled via --tools. The model sees exactly what
 * the flags specify: the system prompt, the loaded context, and the user's
 * prompts. The only thing taken from the environment is the provider API
 * key. rath development itself is meant to happen inside `rath run`.
 *
 * Built on pi-agent-core's agentLoop and pi-ai's provider registry; the
 * openai-native provider is registered, so hosted web search is available
 * (on by default, --no-web-search disables it). Client-side tools are
 * loaded on demand from @earendil-works/pi-coding-agent.
 */
import { readFileSync, writeFileSync } from "node:fs";
import * as readline from "node:readline/promises";
import { type AgentContext, type AgentTool, agentLoop } from "@earendil-works/pi-agent-core";
import {
  type Api,
  getModel,
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

const REASONING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"] as const;
type ReasoningLevel = (typeof REASONING_LEVELS)[number];

const TOOL_NAMES = ["read", "bash", "edit", "write", "grep", "find", "ls"] as const;
type ToolName = (typeof TOOL_NAMES)[number];

interface RunFlags {
  model: string;
  prompt?: string;
  systemPrompt: string;
  reasoning: ReasoningLevel;
  webSearch: boolean;
  tools: ToolName[];
  loadPath?: string;
  savePath?: string;
}

/**
 * Load the requested client-side tools from @earendil-works/pi-coding-agent.
 * The dependency is imported on demand so that rath's published package does
 * not require it; it is only needed when --tools is used.
 */
async function loadTools(names: ToolName[], cwd: string): Promise<AgentTool[]> {
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

function dim(text: string): string {
  return process.stdout.isTTY ? `${DIM}${text}${RESET}` : text;
}

function resolveModel(spec: string): Model<Api> {
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

interface SerializedContext {
  systemPrompt?: string;
  messages: Message[];
}

function loadContext(path: string): SerializedContext {
  const parsed = JSON.parse(readFileSync(path, "utf8")) as SerializedContext;
  if (!Array.isArray(parsed.messages)) {
    throw new Error(`${path} does not contain a serialized context (messages array missing)`);
  }
  return parsed;
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
interface RenderedCitationsBlock {
  type: "text";
  text: string;
  renderedCitations: true;
}

function isRenderedCitations(block: { type: string }): block is RenderedCitationsBlock {
  return block.type === "text" && (block as RenderedCitationsBlock).renderedCitations === true;
}

function renderCitationsTrailer(message: Message & { role: "assistant" }): string | undefined {
  const citations = message.content
    .filter((b): b is TextContent => b.type === "text" && !isRenderedCitations(b))
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
  return `Sources:\n${lines.join("\n")}`;
}

/**
 * Run one agent-loop invocation for `prompts`, rendering events to stdout.
 * Returns false if the run ended in an error message.
 */
async function runPrompt(
  prompts: Message[],
  context: AgentContext,
  model: Model<Api>,
  flags: RunFlags,
): Promise<boolean> {
  const streamFn: typeof streamSimple = (m, ctx, options) =>
    streamSimple(m, ctx, {
      ...options,
      webSearch: flags.webSearch,
    } as SimpleStreamOptions);

  // openai-native replays real annotations; the rendered trailer would
  // duplicate them. Other providers get the trailer as plain text.
  const stripTrailers = model.api === OPENAI_NATIVE_API;
  const events = agentLoop(
    prompts,
    context,
    {
      model,
      ...(flags.reasoning === "off" ? {} : { reasoning: flags.reasoning }),
      convertToLlm: (messages) =>
        messages.flatMap((m): Message[] => {
          if (m.role !== "user" && m.role !== "assistant" && m.role !== "toolResult") {
            return [];
          }
          if (m.role === "assistant" && stripTrailers) {
            return [{ ...m, content: m.content.filter((b) => !isRenderedCitations(b)) }];
          }
          return [m];
        }),
    },
    undefined,
    streamFn,
  );

  let openLine = false;
  const ensureNewline = () => {
    if (openLine) {
      process.stdout.write("\n");
      openLine = false;
    }
  };
  const seenHostedCalls = new Set<string>();
  let ok = true;

  for await (const event of events) {
    if (event.type === "message_update") {
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
          ok = false;
          continue;
        }
        const trailer = renderCitationsTrailer(message);
        if (trailer) {
          // Merge into the transcript (persisted via --save, flattened for
          // foreign providers) and show it.
          const block: RenderedCitationsBlock = {
            type: "text",
            text: trailer,
            renderedCitations: true,
          };
          message.content.push(block);
          process.stdout.write(`${dim(trailer)}\n`);
        }
      }
    }
  }
  // agentLoop treats the context as a snapshot; the caller owns appending
  // the run's new messages (prompts included) to the live context.
  const newMessages = await events.result();
  context.messages.push(
    ...newMessages.filter(
      (m): m is Message =>
        m.role === "user" || m.role === "assistant" || m.role === "toolResult",
    ),
  );
  return ok;
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
    "Without --prompt, reads prompts interactively; /exit or Ctrl+D ends the\n" +
    "session. With --prompt, runs one prompt and exits (0 on success).",
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
    let model: Model<Api>;
    let context: AgentContext;
    try {
      model = resolveModel(flags.model);
      if (systemPromptFile) {
        flags.systemPrompt = readFileSync(systemPromptFile, "utf8").trim();
      }
      const loaded = flags.loadPath ? loadContext(flags.loadPath) : undefined;
      context = {
        systemPrompt: loaded?.systemPrompt ?? flags.systemPrompt,
        messages: loaded?.messages ?? [],
        tools: await loadTools(flags.tools, process.cwd()),
      };
    } catch (error) {
      process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
      return 1;
    }

    const save = () => {
      if (flags.savePath) {
        writeFileSync(
          flags.savePath,
          `${JSON.stringify(
            { systemPrompt: context.systemPrompt, messages: context.messages },
            null,
            2,
          )}\n`,
        );
        process.stderr.write(`${dim(`context saved to ${flags.savePath}`)}\n`);
      }
    };

    if (flags.prompt !== undefined) {
      const ok = await runPrompt(
        [{ role: "user", content: flags.prompt, timestamp: Date.now() }],
        context,
        model,
        flags,
      );
      save();
      return ok ? 0 : 1;
    }

    process.stderr.write(`${dim(`model: ${flags.model} | /exit or Ctrl+D to quit`)}\n`);
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
        if (trimmed === "/exit") {
          break;
        }
        await runPrompt(
          [{ role: "user", content: trimmed, timestamp: Date.now() }],
          context,
          model,
          flags,
        );
      }
    } finally {
      rl.close();
      save();
    }
    return 0;
  },
};
