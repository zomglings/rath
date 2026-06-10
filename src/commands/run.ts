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
import { readFileSync } from "node:fs";
import * as readline from "node:readline/promises";
import { Agent } from "@earendil-works/pi-agent-core";
import { type Message, type SimpleStreamOptions, streamSimple } from "@earendil-works/pi-ai";
import { type Command, fullName, helpRequested, helpText } from "../command.js";
import {
  contentBlocks,
  isHostedToolCall,
  OPENAI_NATIVE_API,
  registerOpenAINative,
} from "../index.js";
import {
  applyCitationTrailer,
  handleSlashCommand,
  isRenderedCitations,
  loadContext,
  loadTools,
  REASONING_LEVELS,
  type ReasoningLevel,
  type RunFlags,
  resolveModel,
  saveContext,
  TOOL_NAMES,
  type ToolName,
} from "./run-shared.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function dim(text: string): string {
  return process.stdout.isTTY ? `${DIM}${text}${RESET}` : text;
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
