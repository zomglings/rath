/**
 * pi-tui frontend for `rath run` (-T/--tui). Same session semantics as the
 * plain REPL — same Agent, same slash commands, same citation merging — with
 * differential rendering, an editor input, and Ctrl+C interrupting the
 * current turn instead of killing the session.
 *
 * Messages stream as plain text and are re-rendered on completion: markdown
 * for assistant prose, colored markers for tool and hosted-tool calls, and
 * OSC 8 hyperlinks for citations.
 */
import type { Agent } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import {
  type Component,
  Container,
  Editor,
  type EditorTheme,
  hyperlink,
  Loader,
  Markdown,
  type MarkdownTheme,
  matchesKey,
  ProcessTerminal,
  type SelectItem,
  SelectList,
  type SelectListTheme,
  Text,
  TUI,
} from "@earendil-works/pi-tui";
import { contentBlocks, getCitations, isHostedToolCall } from "../index.js";
import {
  applyCitationTrailer,
  handleSlashCommand,
  isRenderedCitations,
  listModels,
  REASONING_LEVELS,
  type RunFlags,
  sessionInfo,
} from "./run-shared.js";

// Styles do not carry across TUI lines; every style helper applies per line.
function styled(open: string): (text: string) => string {
  return (text: string) =>
    text
      .split("\n")
      .map((line) => `${open}${line}\x1b[0m`)
      .join("\n");
}

const identity = (s: string) => s;
const dim = styled("\x1b[2m");
const bold = styled("\x1b[1m");
const italic = styled("\x1b[3m");
const underline = styled("\x1b[4m");
const strikethrough = styled("\x1b[9m");
const inverse = styled("\x1b[7m");
const cyan = styled("\x1b[36m");
const yellow = styled("\x1b[33m");
const red = styled("\x1b[31m");
const blueUnderline = styled("\x1b[34m\x1b[4m");
const boldCyan = styled("\x1b[1m\x1b[36m");

const selectTheme: SelectListTheme = {
  selectedPrefix: identity,
  selectedText: inverse,
  description: dim,
  scrollInfo: dim,
  noMatch: dim,
};

const editorTheme: EditorTheme = {
  borderColor: dim,
  selectList: selectTheme,
};

const markdownTheme: MarkdownTheme = {
  heading: boldCyan,
  link: blueUnderline,
  linkUrl: dim,
  code: yellow,
  codeBlock: identity,
  codeBlockBorder: dim,
  quote: italic,
  quoteBorder: dim,
  hr: dim,
  listBullet: cyan,
  bold,
  italic,
  strikethrough,
  underline,
};

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

function truncateArgs(args: unknown): string {
  const text = JSON.stringify(args);
  return text.length > 120 ? `${text.slice(0, 120)}…` : text;
}

/** Plain-text rendering used while a message is streaming. */
function renderStreaming(message: AssistantMessage): string {
  const parts: string[] = [];
  for (const block of contentBlocks(message)) {
    if (block.type === "thinking") {
      if (block.thinking.trim().length > 0) {
        parts.push(dim(block.thinking));
      }
    } else if (isHostedToolCall(block)) {
      parts.push(yellow(`[${block.toolName}]`));
    } else if (block.type === "toolCall") {
      parts.push(yellow(`[${block.name}: ${truncateArgs(block.arguments)}]`));
    } else if (block.type === "text") {
      parts.push(isRenderedCitations(block) ? dim(block.text) : block.text);
    }
  }
  return parts.join("\n");
}

/** Clickable sources list rebuilt from the message's citations. */
function sourcesComponent(message: AssistantMessage): Text | undefined {
  const urls = new Map<string, string>();
  for (const block of message.content) {
    if (block.type === "text" && !isRenderedCitations(block)) {
      for (const citation of getCitations(block)) {
        if (citation.type === "url_citation" && !urls.has(citation.url)) {
          urls.set(citation.url, citation.title);
        }
      }
    }
  }
  if (urls.size === 0) {
    return undefined;
  }
  const lines = [...urls].map(
    ([url, title]) => `  ${hyperlink(title || url, url)}${title ? dim(` — ${url}`) : ""}`,
  );
  return new Text(`${dim("sources:")}\n${lines.join("\n")}`);
}

/** Pretty rendering for a completed assistant message. */
function prettyAssistant(message: AssistantMessage): Component {
  const out = new Container();
  for (const block of contentBlocks(message)) {
    if (block.type === "thinking") {
      if (block.thinking.trim().length > 0) {
        out.addChild(new Text(dim(block.thinking)));
      }
    } else if (isHostedToolCall(block)) {
      out.addChild(new Text(yellow(`[${block.toolName}]`)));
    } else if (block.type === "toolCall") {
      out.addChild(new Text(yellow(`[${block.name}: ${truncateArgs(block.arguments)}]`)));
    } else if (block.type === "text") {
      if (isRenderedCitations(block)) {
        const sources = sourcesComponent(message);
        if (sources) {
          out.addChild(sources);
        }
      } else if (block.text.trim().length > 0) {
        out.addChild(new Markdown(block.text, 0, 0, markdownTheme));
      }
    }
  }
  if (message.stopReason === "error" || message.stopReason === "aborted") {
    out.addChild(new Text(red(`error: ${message.errorMessage ?? message.stopReason}`)));
  }
  return out;
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

function userEcho(text: string): Text {
  return new Text(`\n${cyan("›")} ${bold(text)}`);
}

export async function runTui(agent: Agent, flags: RunFlags): Promise<number> {
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    process.stderr.write("--tui requires an interactive terminal\n");
    return 1;
  }

  const tui = new TUI(new ProcessTerminal());
  const transcript = new Container();
  const status = new Container();
  const loader = new Loader(tui, cyan, dim, "thinking…", { frames: SPINNER, intervalMs: 80 });
  const editor = new Editor(tui, editorTheme);

  const banner = new Text("");
  const refreshBanner = () => {
    banner.setText(
      `${cyan("rath")} ${dim(
        `| ${flags.model} | reasoning: ${agent.state.thinkingLevel} | ` +
          "/info /sys /model /lsmodels /reasoning /exit | Ctrl+C interrupts",
      )}`,
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
      transcript.addChild(new Text(result.isError ? red(result.output) : dim(result.output)));
    }
    refreshBanner();
  };

  /** TUI-native rendering for selected slash commands. Returns true when handled. */
  const handleTuiCommand = (input: string): boolean => {
    const [command = "", ...rest] = input.split(/\s+/);
    const arg = rest.join(" ").trim();
    if (command === "/model" && arg.length === 0) {
      const specs = listModels();
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
      const panel = pairs.map(([key, value]) => `${dim(key.padEnd(width))}  ${value}`).join("\n");
      transcript.addChild(new Text(panel, 2, 1));
      return true;
    }
    return false;
  };

  // Replay a loaded context into the transcript so the session is visible.
  for (const message of agent.state.messages) {
    if (message.role === "user") {
      transcript.addChild(userEcho(messageText(message.content)));
    } else if (message.role === "assistant") {
      transcript.addChild(prettyAssistant(message as AssistantMessage));
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
        transcript.addChild(userEcho(messageText(event.message.content)));
      } else if (event.message.role === "assistant") {
        current = new Text("");
        transcript.addChild(current);
      }
    } else if (event.type === "message_update") {
      if (event.message.role === "assistant") {
        current?.setText(renderStreaming(event.message as AssistantMessage));
      }
    } else if (event.type === "message_end") {
      if (event.message.role === "assistant") {
        const message = event.message as AssistantMessage;
        if (message.stopReason !== "error" && message.stopReason !== "aborted") {
          applyCitationTrailer(message);
        }
        // Swap the streaming text for the pretty rendering.
        if (current) {
          transcript.removeChild(current);
          current = null;
        }
        transcript.addChild(prettyAssistant(message));
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
      transcript.addChild(userEcho(trimmed));
      if (!handleTuiCommand(trimmed)) {
        const result = handleSlashCommand(trimmed, agent, flags);
        if (result) {
          if (result.output) {
            transcript.addChild(new Text(result.isError ? red(result.output) : dim(result.output)));
          }
          refreshBanner();
          if (result.exit) {
            tui.stop();
            resolveDone(0);
          }
        }
      }
      tui.requestRender();
      return;
    }
    const message = { role: "user" as const, content: trimmed, timestamp: Date.now() };
    if (agent.state.isStreaming) {
      // Injected after the current turn finishes; the loop emits its
      // message_start when it lands.
      agent.steer(message);
      transcript.addChild(new Text(dim(`(queued) › ${trimmed}`)));
      tui.requestRender();
      return;
    }
    agent.prompt(message).catch((error) => {
      transcript.addChild(
        new Text(red(`error: ${error instanceof Error ? error.message : error}`)),
      );
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
