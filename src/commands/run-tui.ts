/**
 * pi-tui frontend for `rath run` (-T/--tui). Same session semantics as the
 * plain REPL — same Agent, same slash commands, same citation merging — with
 * differential rendering, an editor input, and Ctrl+C interrupting the
 * current turn instead of killing the session.
 */
import type { Agent } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import {
  Container,
  Editor,
  type EditorTheme,
  Loader,
  matchesKey,
  ProcessTerminal,
  type SelectItem,
  SelectList,
  type SelectListTheme,
  Text,
  TUI,
} from "@earendil-works/pi-tui";
import { contentBlocks, isHostedToolCall } from "../index.js";
import {
  applyCitationTrailer,
  handleSlashCommand,
  isRenderedCitations,
  listModels,
  REASONING_LEVELS,
  type RunFlags,
  sessionInfo,
} from "./run-shared.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";
const INVERSE = "\x1b[7m";
const UNINVERSE = "\x1b[27m";

/** Styles do not carry across TUI lines; apply dim per line. */
function dimLines(text: string): string {
  return text
    .split("\n")
    .map((line) => `${DIM}${line}${RESET}`)
    .join("\n");
}

const identity = (s: string) => s;
const dimColor = (s: string) => `${DIM}${s}${RESET}`;
const inverse = (s: string) => `${INVERSE}${s}${UNINVERSE}`;

const selectTheme: SelectListTheme = {
  selectedPrefix: identity,
  selectedText: inverse,
  description: dimColor,
  scrollInfo: dimColor,
  noMatch: dimColor,
};

const editorTheme: EditorTheme = {
  borderColor: dimColor,
  selectList: selectTheme,
};

function renderAssistant(message: AssistantMessage): string {
  const parts: string[] = [];
  for (const block of contentBlocks(message)) {
    if (block.type === "thinking") {
      if (block.thinking.trim().length > 0) {
        parts.push(dimLines(block.thinking));
      }
    } else if (isHostedToolCall(block)) {
      parts.push(dimLines(`[${block.toolName}]`));
    } else if (block.type === "toolCall") {
      const args = JSON.stringify(block.arguments);
      parts.push(
        dimLines(`[${block.name}: ${args.length > 120 ? `${args.slice(0, 120)}…` : args}]`),
      );
    } else if (block.type === "text") {
      parts.push(isRenderedCitations(block) ? dimLines(block.text) : block.text);
    }
  }
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    parts.push(`error: ${message.errorMessage ?? message.stopReason}`);
  }
  return parts.join("\n");
}

function messageText(content: string | { type: string; text?: string }[]): string {
  if (typeof content === "string") {
    return content;
  }
  return content
    .filter((c) => c.type === "text")
    .map((c) => c.text ?? "")
    .join("\n");
}

export async function runTui(agent: Agent, flags: RunFlags): Promise<number> {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    process.stderr.write("--tui requires an interactive terminal\n");
    return 1;
  }

  const tui = new TUI(new ProcessTerminal());
  const transcript = new Container();
  const status = new Container();
  const loader = new Loader(tui, dimColor, dimColor, "working");
  const editor = new Editor(tui, editorTheme);

  const banner = new Text("");
  const refreshBanner = () => {
    banner.setText(
      dimLines(
        `model: ${flags.model} | reasoning: ${agent.state.thinkingLevel} | ` +
          "/info /model /lsmodels /reasoning /exit | Ctrl+C interrupts",
      ),
    );
  };
  refreshBanner();
  tui.addChild(banner);
  tui.addChild(transcript);
  tui.addChild(status);
  tui.addChild(editor);
  tui.setFocus(editor);

  /** Overlay selector; Enter applies, Escape cancels. */
  const openSelector = (items: SelectItem[], onPick: (value: string) => void) => {
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

  const echoCommandResult = (input: string) => {
    const result = handleSlashCommand(input, agent, flags);
    if (result?.output) {
      transcript.addChild(new Text(result.isError ? result.output : dimLines(result.output)));
    }
    refreshBanner();
  };

  /** TUI-native rendering for selected slash commands. Returns true when handled. */
  const handleTuiCommand = (input: string): boolean => {
    const [command = "", ...rest] = input.split(/\s+/);
    const arg = rest.join(" ").trim();
    if (command === "/lsmodels" || (command === "/model" && arg.length === 0)) {
      const specs = listModels(arg.length > 0 ? arg : undefined);
      if (specs.length === 0) {
        transcript.addChild(new Text(dimLines(`no models matching "${arg}"`)));
        return true;
      }
      openSelector(
        specs.map((spec) => ({ value: spec, label: spec })),
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
        .map(([key, value]) => `${dimColor(key.padEnd(width))}  ${value}`)
        .join("\n");
      transcript.addChild(new Text(panel, 2, 1));
      return true;
    }
    return false;
  };

  // Replay a loaded context into the transcript so the session is visible.
  for (const message of agent.state.messages) {
    if (message.role === "user") {
      transcript.addChild(new Text(`\n> ${messageText(message.content)}`));
    } else if (message.role === "assistant") {
      transcript.addChild(new Text(renderAssistant(message as AssistantMessage)));
    }
  }

  let resolveDone: (code: number) => void = () => {};
  const done = new Promise<number>((resolve) => {
    resolveDone = resolve;
  });

  let current: Text | null = null;
  agent.subscribe((event) => {
    if (event.type === "agent_start") {
      status.addChild(loader);
      loader.start();
    } else if (event.type === "agent_end") {
      loader.stop();
      status.clear();
    } else if (event.type === "message_start") {
      if (event.message.role === "user") {
        transcript.addChild(new Text(`\n> ${messageText(event.message.content)}`));
      } else if (event.message.role === "assistant") {
        current = new Text("");
        transcript.addChild(current);
      }
    } else if (event.type === "message_update") {
      if (event.message.role === "assistant") {
        current?.setText(renderAssistant(event.message as AssistantMessage));
      }
    } else if (event.type === "message_end") {
      if (event.message.role === "assistant") {
        const message = event.message as AssistantMessage;
        if (message.stopReason !== "error" && message.stopReason !== "aborted") {
          applyCitationTrailer(message);
        }
        current?.setText(renderAssistant(message));
        current = null;
      }
    }
    tui.requestRender();
  });

  editor.onSubmit = (text: string) => {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return;
    }
    if (trimmed.startsWith("/")) {
      transcript.addChild(new Text(`\n> ${trimmed}`));
      if (handleTuiCommand(trimmed)) {
        tui.requestRender();
        return;
      }
      const result = handleSlashCommand(trimmed, agent, flags);
      if (result) {
        if (result.output) {
          transcript.addChild(new Text(result.isError ? result.output : dimLines(result.output)));
        }
        refreshBanner();
        if (result.exit) {
          tui.stop();
          resolveDone(0);
        }
        tui.requestRender();
        return;
      }
      tui.requestRender();
      return;
    }
    const message = { role: "user" as const, content: trimmed, timestamp: Date.now() };
    if (agent.state.isStreaming) {
      // Injected after the current turn finishes; the loop emits its
      // message_start when it lands.
      agent.steer(message);
      transcript.addChild(new Text(dimLines(`(queued) > ${trimmed}`)));
      tui.requestRender();
      return;
    }
    agent.prompt(message).catch((error) => {
      transcript.addChild(new Text(`error: ${error instanceof Error ? error.message : error}`));
      tui.requestRender();
    });
  };

  tui.addInputListener((data) => {
    if (matchesKey(data, "ctrl+c")) {
      if (agent.state.isStreaming) {
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
