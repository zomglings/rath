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
 * (default, also used for -p one-shots) and a pi-tui interface (-T/--tui)
 * rendered with Pi's own interactive components. The TUI's dependencies
 * (pi-tui and pi-coding-agent's UI) are imported inside runTui, so they only
 * load when --tui is used; the type-only imports below are erased at compile
 * time and cost nothing.
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
} from "@earendil-works/pi-ai";
import type * as PiCodingAgent from "@earendil-works/pi-coding-agent";
import type * as PiTui from "@earendil-works/pi-tui";
import { type Command, fullName, helpText } from "../command.js";
import {
  applyCitationTrailer,
  contentBlocks,
  flattenHostedContent,
  isHostedToolCall,
  OPENAI_NATIVE_API,
  openaiNativeModel,
  registerOpenAINative,
  stripRenderedCitations,
  uniqueUrlCitations,
} from "../index.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

// dim() wraps status text, which the plain frontend writes to stderr; gate the
// escape codes on stderr's TTY status so piping stdout keeps the dimming and
// redirecting stderr to a file does not capture raw escapes.
function dim(text: string): string {
  return process.stderr.isTTY ? `${DIM}${text}${RESET}` : text;
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
  // Typed as AgentTool so pi-coding-agent API drift fails compilation here.
  const factories: Record<ToolName, (cwd: string) => AgentTool> = {
    read: pi.createReadTool,
    bash: pi.createBashTool,
    edit: pi.createEditTool,
    write: pi.createWriteTool,
    grep: pi.createGrepTool,
    find: pi.createFindTool,
    ls: pi.createLsTool,
  };
  return names.map((name) => factories[name](cwd));
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
    (m): m is Message =>
      m.role === "user" ||
      m.role === "toolResult" ||
      // Skip failed turns so they do not accumulate as empty bubbles across
      // saved sessions or replay as malformed items on --load.
      (m.role === "assistant" && m.stopReason !== "error" && m.stopReason !== "aborted"),
  );
  writeFileSync(
    path,
    `${JSON.stringify({ systemPrompt: agent.state.systemPrompt, messages }, null, 2)}\n`,
  );
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
    ["system prompt", "/sys to view or set"],
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
export async function handleSlashCommand(
  input: string,
  agent: Agent,
  flags: RunFlags,
): Promise<SlashResult | undefined> {
  if (!input.startsWith("/")) {
    return undefined;
  }
  const [command = "", ...rest] = input.split(/\s+/);
  const arg = rest.join(" ").trim();
  // Most settings take effect next turn; the running turn already snapshotted.
  const deferred = agent.state.isStreaming ? " (applies after the current turn)" : "";
  switch (command) {
    case "/exit":
      return { exit: true };
    case "/info":
      return {
        output: sessionInfo(agent, flags)
          .map(([key, value]) => `${key}: ${value}`)
          .join("\n"),
      };
    case "/sys": {
      if (arg.length === 0) {
        const prompt = agent.state.systemPrompt;
        return { output: prompt.length > 0 ? prompt : "(empty)" };
      }
      agent.state.systemPrompt = arg;
      flags.systemPrompt = arg;
      return { output: `system prompt set (${arg.length} chars)${deferred}` };
    }
    case "/websearch": {
      if (arg.length === 0) {
        return { output: `web search: ${flags.webSearch ? "on" : "off"}` };
      }
      if (arg !== "on" && arg !== "off") {
        return { output: "Usage: /websearch [on|off]", isError: true };
      }
      flags.webSearch = arg === "on";
      return { output: `web search: ${arg} (openai-native only)${deferred}` };
    }
    case "/tools": {
      if (arg.length === 0) {
        const names = agent.state.tools.map((t) => t.name);
        return { output: `tools: ${names.length > 0 ? names.join(", ") : "none"}` };
      }
      if (arg === "none") {
        agent.state.tools = [];
        flags.tools = [];
        return { output: `tools: none${deferred}` };
      }
      const requested = arg
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);
      const invalid = requested.filter((n) => !TOOL_NAMES.includes(n as ToolName));
      if (invalid.length > 0) {
        return {
          output: `Unknown tool(s): ${invalid.join(", ")} (use ${TOOL_NAMES.join(", ")})`,
          isError: true,
        };
      }
      const names = requested as ToolName[];
      try {
        agent.state.tools = await loadTools(names, process.cwd());
        flags.tools = names;
        return { output: `tools: ${names.join(", ")}${deferred}` };
      } catch (error) {
        return { output: error instanceof Error ? error.message : String(error), isError: true };
      }
    }
    case "/save": {
      const path = arg.length > 0 ? arg : flags.savePath;
      if (!path) {
        return { output: "Usage: /save <path> (also sets the save-on-exit path)", isError: true };
      }
      try {
        saveContext(agent, path);
        flags.savePath = path;
        return { output: `saved to ${path}` };
      } catch (error) {
        return { output: error instanceof Error ? error.message : String(error), isError: true };
      }
    }
    case "/model": {
      if (arg.length === 0) {
        return { output: `model: ${flags.model}` };
      }
      try {
        agent.state.model = resolveModel(arg);
        flags.model = arg;
        return { output: `model: ${arg}${deferred}` };
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
      // "off" is honored as-is (reasoning disabled), not clamped up to a
      // supported level — only positive levels outside the set are clamped.
      const clampNote =
        arg === "off" || supported.includes(arg as ReasoningLevel)
          ? ""
          : ` (outside ${flags.model}'s supported levels — openai-native clamps it)`;
      return { output: `reasoning: ${arg}${clampNote}${deferred}` };
    }
    default:
      return {
        output:
          `Unknown command: ${command} (commands: /info, /sys [text], /model [spec], ` +
          "/lsmodels [filter], /reasoning [level], /websearch [on|off], /tools [names|none], " +
          "/save [path], /exit)",
        isError: true,
      };
  }
}

interface Renderer {
  /** True when the most recent run produced an error message. */
  hadError: () => boolean;
}

/**
 * Wire plain rendering onto the agent's event stream. The model's answer text
 * goes to stdout (the deliverable, so `-p > file` captures just that);
 * thinking, hosted-tool markers, tool-execution lines, and errors go to
 * stderr as status.
 */
function attachRenderer(agent: Agent): Renderer {
  // Track which stream has an unterminated line so the next block starts fresh.
  let openStream: NodeJS.WriteStream | null = null;
  const write = (stream: NodeJS.WriteStream, text: string) => {
    stream.write(text);
    openStream = text.endsWith("\n") ? null : stream;
  };
  const ensureNewline = () => {
    if (openStream) {
      openStream.write("\n");
      openStream = null;
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
        write(process.stdout, e.delta);
      } else if (e.type === "thinking_delta") {
        write(process.stderr, dim(e.delta));
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
            write(process.stderr, `${dim(`[${block.toolName}]`)}\n`);
          }
        }
      }
    } else if (event.type === "tool_execution_start") {
      ensureNewline();
      const args = JSON.stringify(event.args);
      write(
        process.stderr,
        `${dim(`[${event.toolName}: ${args.length > 120 ? `${args.slice(0, 120)}…` : args}]`)}\n`,
      );
    } else if (event.type === "message_end") {
      const message = event.message;
      if (message.role === "assistant") {
        ensureNewline();
        if (message.stopReason === "error" || message.stopReason === "aborted") {
          write(
            process.stderr,
            `${message.stopReason}: ${message.errorMessage ?? "interrupted"}\n`,
          );
          // A user-initiated interrupt is not a failure; only a genuine error
          // makes the session exit non-zero.
          if (message.stopReason === "error") {
            errored = true;
          }
          return;
        }
        const trailer = applyCitationTrailer(message);
        if (trailer) {
          write(process.stderr, `${dim(trailer)}\n`);
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
    "with -T/--tui). Every startup setting is also settable in-session:\n" +
    "  /info                  show the session configuration\n" +
    "  /sys [text]            show or set the system prompt\n" +
    "  /model [provider/id]   show or switch the model\n" +
    "  /lsmodels [filter]     list available models\n" +
    "  /reasoning [level]     show or set the reasoning level\n" +
    "  /websearch [on|off]    show or toggle hosted web search\n" +
    "  /tools [names|none]    show or set client-side tools\n" +
    "  /save [path]           write the context now and save there on exit\n" +
    "  /exit                  quit\n" +
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
    const flags: RunFlags = {
      model: `${OPENAI_NATIVE_API}/gpt-5-mini`,
      systemPrompt: "You are a helpful assistant.",
      reasoning: "low",
      webSearch: true,
      tools: [],
      tui: false,
    };
    let systemPromptFile: string | undefined;
    let systemPromptExplicit = false;
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
        // Help is honored only as a standalone token, so a flag value of
        // "-h"/"--help" (e.g. --prompt --help) is taken literally.
        if (token === "-h" || token === "--help") {
          process.stdout.write(`${helpText(this, prefix)}\n`);
          return 0;
        }
        if (token === "-m" || token === "--model") {
          flags.model = value();
        } else if (token === "-p" || token === "--prompt") {
          flags.prompt = value();
        } else if (token === "-T" || token === "--tui") {
          flags.tui = true;
        } else if (token === "--system-prompt") {
          flags.systemPrompt = value();
          systemPromptExplicit = true;
        } else if (token === "--system-prompt-file") {
          systemPromptFile = value();
          systemPromptExplicit = true;
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

    if (flags.prompt !== undefined && flags.tui) {
      process.stderr.write("--tui is ignored with --prompt (one-shot, non-interactive)\n");
    }

    registerOpenAINative();
    let agent: Agent;
    try {
      const model = resolveModel(flags.model);
      if (systemPromptFile) {
        flags.systemPrompt = readFileSync(systemPromptFile, "utf8").trim();
      }
      const loaded = flags.loadPath ? loadContext(flags.loadPath) : undefined;
      // A loaded context supplies the system prompt unless a flag overrides it.
      const systemPrompt = systemPromptExplicit
        ? flags.systemPrompt
        : (loaded?.systemPrompt ?? flags.systemPrompt);
      agent = new Agent({
        initialState: {
          systemPrompt,
          model,
          thinkingLevel: flags.reasoning,
          messages: loaded?.messages ?? [],
          tools: flags.tools.length > 0 ? await loadTools(flags.tools, process.cwd()) : [],
        },
        streamFn: (m, ctx, options) =>
          streamSimple(m, ctx, {
            ...options,
            webSearch: flags.webSearch,
          } as SimpleStreamOptions),
        convertToLlm: (messages) => {
          // Failed turns are dropped below; their tool results must be dropped
          // too, or a function_call_output is left orphaned and the API
          // rejects every subsequent request.
          const droppedToolCallIds = new Set<string>();
          for (const m of messages) {
            if (
              m.role === "assistant" &&
              (m.stopReason === "error" || m.stopReason === "aborted")
            ) {
              for (const b of m.content) {
                if (b.type === "toolCall") {
                  droppedToolCallIds.add(b.id);
                }
              }
            }
          }
          return messages.flatMap((m): Message[] => {
            if (m.role === "toolResult") {
              return droppedToolCallIds.has(m.toolCallId) ? [] : [m];
            }
            if (m.role === "user") {
              return [m];
            }
            if (m.role !== "assistant") {
              return [];
            }
            // Drop failed turns (empty content, unreplayable state) so a
            // transient error does not poison every later request.
            if (m.stopReason === "error" || m.stopReason === "aborted") {
              return [];
            }
            // Replay is lossless only to the EXACT producing model: the
            // provider converters key replayability on api+provider+model
            // (encrypted reasoning and hosted raw items are model-specific),
            // so two models sharing an api (e.g. openai-native/gpt-5-mini and
            // openai-native/gpt-5) are NOT interchangeable. Match that triple
            // here, or a cross-model switch would skip flatten and the
            // converter would then silently drop the hosted history anyway.
            const sameModel =
              m.api === agent.state.model.api &&
              m.provider === agent.state.model.provider &&
              m.model === agent.state.model.id;
            if (sameModel) {
              // The rendered citation trailer is a display/persistence artifact;
              // the producing provider reconstructs the real annotations itself,
              // so strip the trailer before replay. (For openai-native the
              // provider's converter also strips it defensively; doing it here
              // keeps the wire payload identical for same-model replay.)
              return [stripRenderedCitations(m)];
            }
            // Handoff to a different model (foreign provider, or another model
            // of the same provider): the target converter only understands
            // text | thinking | toolCall and would silently drop hostedToolCall
            // blocks, ignore inline citations, and reject the producing model's
            // thinking signatures. Flatten the extended content to plain text
            // the target preserves.
            return [flattenHostedContent(m)];
          });
        },
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
      const code = await runTui(agent, flags);
      save();
      return code;
    }

    const renderer = attachRenderer(agent);
    process.stderr.write(
      `${dim(`model: ${flags.model} | /info for commands | Ctrl+C interrupts, Ctrl+D quits`)}\n`,
    );
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    // Iterating the interface (rather than looping rl.question) queues input
    // arriving while a turn runs instead of dropping it. Ctrl+C aborts the
    // active response; a second Ctrl+C before it settles (or one while idle)
    // quits. Note: abort stops the in-flight model request, but an
    // already-running tool finishes, and one queued follow-up turn may still
    // start — the agent loop only observes the abort at request boundaries.
    let abortRequested = false;
    rl.on("SIGINT", () => {
      if (agent.state.isStreaming && !abortRequested) {
        abortRequested = true;
        agent.abort();
      } else {
        rl.close();
      }
    });
    agent.subscribe((event) => {
      if (event.type === "agent_start") {
        abortRequested = false;
      }
    });
    process.stderr.write("rath> ");
    try {
      for await (const line of rl) {
        const trimmed = line.trim();
        if (trimmed.length === 0) {
          process.stderr.write("rath> ");
          continue;
        }
        const result = await handleSlashCommand(trimmed, agent, flags);
        if (result) {
          if (result.output !== undefined) {
            const text = result.output.length > 0 ? result.output : "(empty)";
            process.stderr.write(`${result.isError ? text : dim(text)}\n`);
          }
          if (result.exit) {
            break;
          }
          process.stderr.write("rath> ");
          continue;
        }
        await agent.prompt(trimmed);
        process.stderr.write("rath> ");
      }
    } finally {
      rl.close();
      save();
    }
    return renderer.hadError() ? 1 : 0;
  },
};

// ---------------------------------------------------------------------------
// TUI frontend (-T/--tui)
// ---------------------------------------------------------------------------

// Styles do not carry across TUI lines; every style helper applies per line.
function styled(open: string): (text: string) => string {
  return (text: string) =>
    text
      .split("\n")
      .map((line) => `${open}${line}\x1b[0m`)
      .join("\n");
}

const dimLine = styled("\x1b[2m");
const cyanLine = styled("\x1b[36m");
const yellowLine = styled("\x1b[33m");
const redLine = styled("\x1b[31m");

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function messageText(content: string | { type: string; text?: string }[]): string {
  if (typeof content === "string") {
    return content;
  }
  return content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("\n");
}

/**
 * pi-tui frontend. Same session semantics as the plain REPL — same Agent,
 * same slash commands, same citation merging — rendered with Pi's own
 * interactive components (AssistantMessageComponent, UserMessageComponent,
 * ToolExecutionComponent, and the Pi theme), so the TUI looks like vanilla
 * Pi. rath adds what those components do not know about: hosted-tool
 * markers and clickable citation sources. Messages stream as they arrive
 * and are re-rendered in full on completion.
 */
async function runTui(agent: Agent, flags: RunFlags): Promise<number> {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    process.stderr.write("--tui requires an interactive terminal\n");
    return 1;
  }

  // The TUI's dependencies load here so plain runs never pay for them.
  const piTui = await import("@earendil-works/pi-tui");
  const piCa = await import("@earendil-works/pi-coding-agent");
  const {
    Container,
    Editor,
    hyperlink,
    Loader,
    matchesKey,
    ProcessTerminal,
    SelectList,
    Text,
    TUI,
  } = piTui;
  const {
    AssistantMessageComponent,
    getMarkdownTheme,
    getSelectListTheme,
    initTheme,
    ToolExecutionComponent,
    UserMessageComponent,
  } = piCa;

  // Pi's built-in dark theme, selected explicitly. Caveat: initTheme also
  // registers (not applies) themes found in ~/.pi/agent/themes — the only
  // implicit read in rath run, cosmetic-only, forced on us because
  // pi-coding-agent does not export setThemeInstance/loadThemeFromPath.
  initTheme("dark", false);
  const markdownTheme = getMarkdownTheme();
  const selectTheme = getSelectListTheme();
  const editorTheme: PiTui.EditorTheme = { borderColor: dimLine, selectList: selectTheme };

  /** Clickable sources list rebuilt from the message's citations. */
  const sourcesComponent = (message: AssistantMessage): PiTui.Text | undefined => {
    const citations = uniqueUrlCitations(message);
    if (citations.length === 0) {
      return undefined;
    }
    const lines = citations.map(
      (c) => `  ${hyperlink(c.title || c.url, c.url)}${c.title ? dimLine(` — ${c.url}`) : ""}`,
    );
    return new Text(`${dimLine("sources:")}\n${lines.join("\n")}`);
  };

  const hostedMarkers = (message: AssistantMessage): PiTui.Text[] =>
    contentBlocks(message)
      .filter(isHostedToolCall)
      .map((block) => new Text(yellowLine(`[${block.toolName}]`)));

  const tui = new TUI(new ProcessTerminal());
  const transcript = new Container();
  const status = new Container();
  const loader = new Loader(tui, cyanLine, dimLine, "thinking…", {
    frames: SPINNER,
    intervalMs: 80,
  });
  const editor = new Editor(tui, editorTheme);

  const banner = new Text("");
  const refreshBanner = () => {
    banner.setText(
      `${cyanLine("rath")} ${dimLine(
        `| ${flags.model} | reasoning: ${agent.state.thinkingLevel} | ` +
          "/info for commands | Ctrl+C interrupts",
      )}`,
    );
  };
  refreshBanner();
  tui.addChild(banner);
  tui.addChild(transcript);
  tui.addChild(status);
  tui.addChild(editor);
  tui.setFocus(editor);

  const addAssistant = (message: AssistantMessage) => {
    transcript.addChild(
      new AssistantMessageComponent(stripRenderedCitations(message), false, markdownTheme),
    );
    for (const marker of hostedMarkers(message)) {
      transcript.addChild(marker);
    }
    const sources = sourcesComponent(message);
    if (sources) {
      transcript.addChild(sources);
    }
  };

  /** Overlay selector; Enter applies, Escape cancels. */
  const openSelector = (items: PiTui.SelectItem[], onPick: (value: string) => void) => {
    const list = new SelectList(items, 12, selectTheme);
    const handle = tui.showOverlay(list, { width: "80%", maxHeight: "60%" });
    list.onSelect = (item) => {
      handle.hide();
      onPick(item.value);
      tui.requestRender();
    };
    list.onCancel = () => {
      handle.hide();
      tui.requestRender();
    };
  };

  const showCommandOutput = (result: { output?: string; isError?: boolean }) => {
    if (result.output !== undefined) {
      const text = result.output.length > 0 ? result.output : "(empty)";
      transcript.addChild(new Text(result.isError ? redLine(text) : dimLine(text)));
    }
  };
  const echoCommandResult = async (input: string) => {
    showCommandOutput((await handleSlashCommand(input, agent, flags)) ?? {});
    refreshBanner();
    tui.requestRender();
  };

  /** TUI-native rendering for selected slash commands. Returns true when handled. */
  const handleTuiCommand = (input: string): boolean => {
    const [command = "", ...rest] = input.split(/\s+/);
    const arg = rest.join(" ").trim();
    if (command === "/model" && arg.length === 0) {
      openSelector(
        listModels().map((spec) => ({ value: spec, label: spec })),
        (spec) => echoCommandResult(`/model ${spec}`),
      );
      return true;
    }
    if (command === "/reasoning" && arg.length === 0) {
      openSelector(
        REASONING_LEVELS.map((level) => ({
          value: level,
          label: level,
          description: level === agent.state.thinkingLevel ? "current" : "",
        })),
        (level) => echoCommandResult(`/reasoning ${level}`),
      );
      return true;
    }
    if (command === "/info") {
      const pairs = sessionInfo(agent, flags);
      const width = Math.max(...pairs.map(([key]) => key.length));
      const panel = pairs
        .map(([key, value]) => `${dimLine(key.padEnd(width))}  ${value}`)
        .join("\n");
      transcript.addChild(new Text(panel, 2, 1));
      return true;
    }
    return false;
  };

  // Replay a loaded context into the transcript so the session is visible.
  for (const message of agent.state.messages) {
    if (message.role === "user") {
      transcript.addChild(new UserMessageComponent(messageText(message.content), markdownTheme));
    } else if (message.role === "assistant") {
      addAssistant(message as AssistantMessage);
    }
  }

  let resolveDone: (code: number) => void = () => {};
  const done = new Promise<number>((resolve) => {
    resolveDone = resolve;
  });

  let current: PiCodingAgent.AssistantMessageComponent | null = null;
  const toolComponents = new Map<string, PiCodingAgent.ToolExecutionComponent>();
  agent.subscribe((event) => {
    if (event.type === "agent_start") {
      status.addChild(loader);
      loader.start();
    } else if (event.type === "agent_end") {
      loader.stop();
      status.clear();
      toolComponents.clear();
      // A prompt submitted during the final steering-poll window stays queued.
      // Drain it once the run has fully settled (continue() throws while the
      // run is still active), instead of waiting for the next manual prompt.
      if (agent.hasQueuedMessages()) {
        setImmediate(() => {
          if (!agent.state.isStreaming && agent.hasQueuedMessages()) {
            agent.continue().catch((error) => {
              transcript.addChild(
                new Text(redLine(`error: ${error instanceof Error ? error.message : error}`)),
              );
              tui.requestRender();
            });
          }
        });
      }
    } else if (event.type === "message_start") {
      if (event.message.role === "user") {
        transcript.addChild(
          new UserMessageComponent(messageText(event.message.content), markdownTheme),
        );
      } else if (event.message.role === "assistant") {
        current = new AssistantMessageComponent(undefined, false, markdownTheme);
        transcript.addChild(current);
      }
    } else if (event.type === "message_update") {
      if (event.message.role === "assistant") {
        current?.updateContent(stripRenderedCitations(event.message as AssistantMessage));
      }
    } else if (event.type === "message_end") {
      if (event.message.role === "assistant") {
        const message = event.message as AssistantMessage;
        const failed = message.stopReason === "error" || message.stopReason === "aborted";
        if (!failed) {
          applyCitationTrailer(message);
        }
        if (current) {
          // Replace the streaming component with the full final rendering
          // (markers and sources included).
          transcript.removeChild(current);
          current = null;
        }
        // pi-agent-core resolves prompt() normally on provider errors and
        // emits them here, not as a rejection — render the error explicitly
        // instead of an empty assistant bubble.
        if (failed) {
          transcript.addChild(new Text(redLine(`error: ${message.errorMessage ?? "unknown"}`)));
        } else {
          addAssistant(message);
        }
      }
    } else if (event.type === "tool_execution_start") {
      const component = new ToolExecutionComponent(
        event.toolName,
        event.toolCallId,
        event.args,
        undefined,
        undefined,
        tui,
        process.cwd(),
      );
      component.setArgsComplete();
      component.markExecutionStarted();
      toolComponents.set(event.toolCallId, component);
      transcript.addChild(component);
    } else if (event.type === "tool_execution_update") {
      const component = toolComponents.get(event.toolCallId);
      const partial = event.partialResult as
        | { content?: { type: string }[]; details?: unknown }
        | undefined;
      if (component && partial?.content) {
        component.updateResult(
          { content: partial.content, details: partial.details, isError: false },
          true,
        );
      }
    } else if (event.type === "tool_execution_end") {
      const component = toolComponents.get(event.toolCallId);
      const result = event.result as { content?: { type: string }[]; details?: unknown };
      component?.updateResult(
        { content: result?.content ?? [], details: result?.details, isError: event.isError },
        false,
      );
    }
    tui.requestRender();
  });

  editor.onSubmit = (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return;
    }
    if (trimmed.startsWith("/")) {
      transcript.addChild(new Text(`\n${cyanLine("›")} ${trimmed}`));
      if (!handleTuiCommand(trimmed)) {
        handleSlashCommand(trimmed, agent, flags).then((result) => {
          if (result) {
            showCommandOutput(result);
            refreshBanner();
            if (result.exit) {
              tui.stop();
              resolveDone(0);
            }
          }
          tui.requestRender();
        });
      }
      tui.requestRender();
      return;
    }
    const message = { role: "user" as const, content: trimmed, timestamp: Date.now() };
    // Steer while a turn is streaming, or when a message is already queued for
    // the post-run drain — prompting directly in the latter case would jump
    // ahead of the queued one and break submission order.
    if (agent.state.isStreaming || agent.hasQueuedMessages()) {
      agent.steer(message);
      transcript.addChild(new Text(dimLine(`(queued) › ${trimmed}`)));
      tui.requestRender();
      return;
    }
    agent.prompt(message).catch((error) => {
      transcript.addChild(
        new Text(redLine(`error: ${error instanceof Error ? error.message : error}`)),
      );
      tui.requestRender();
    });
  };

  tui.addInputListener((data) => {
    if (matchesKey(data, "ctrl+c")) {
      if (agent.state.isStreaming) {
        // Abort the active response and discard any queued steers, so an
        // interrupt does not surface a forgotten follow-up afterward.
        agent.clearAllQueues();
        agent.abort();
      } else {
        tui.stop();
        resolveDone(0);
      }
      return { consume: true };
    }
    return undefined;
  });

  tui.start();
  const code = await done;
  agent.abort();
  await agent.waitForIdle();
  return code;
}
