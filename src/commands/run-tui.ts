/**
 * pi-tui frontend for `rath run` (-T/--tui). Same session semantics as the
 * plain REPL — same Agent, same slash commands, same citation merging — with
 * differential rendering, an editor input, and Ctrl+C interrupting the
 * current turn instead of killing the session.
 *
 * Rendering reuses Pi's own interactive components from
 * @earendil-works/pi-coding-agent (AssistantMessageComponent,
 * UserMessageComponent, ToolExecutionComponent, and the Pi theme), so the
 * TUI looks like vanilla Pi. rath adds what those components do not know
 * about: hosted-tool markers and clickable citation sources.
 */
import type { Agent } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import {
  AssistantMessageComponent,
  getMarkdownTheme,
  getSelectListTheme,
  initTheme,
  ToolExecutionComponent,
  UserMessageComponent,
} from "@earendil-works/pi-coding-agent";
import {
  type Component,
  Container,
  Editor,
  type EditorTheme,
  hyperlink,
  Loader,
  matchesKey,
  ProcessTerminal,
  type SelectItem,
  SelectList,
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
} from "./run.js";

// Styles do not carry across TUI lines; every style helper applies per line.
function styled(open: string): (text: string) => string {
  return (text: string) =>
    text
      .split("\n")
      .map((line) => `${open}${line}\x1b[0m`)
      .join("\n");
}

const dim = styled("\x1b[2m");
const cyan = styled("\x1b[36m");
const yellow = styled("\x1b[33m");
const red = styled("\x1b[31m");

const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

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

/**
 * Display copy of an assistant message without the rendered-citations
 * trailer: the trailer exists for persistence and provider handoff; the TUI
 * shows clickable sources instead.
 */
function displayMessage(message: AssistantMessage): AssistantMessage {
  return { ...message, content: message.content.filter((b) => !isRenderedCitations(b)) };
}

function hostedMarkers(message: AssistantMessage): Component[] {
  return contentBlocks(message)
    .filter(isHostedToolCall)
    .map((block) => new Text(yellow(`[${block.toolName}]`)));
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

  // Pi's built-in dark theme, selected explicitly. Caveat: initTheme also
  // registers (not applies) themes found in ~/.pi/agent/themes — the only
  // implicit read in rath run, cosmetic-only, forced on us because
  // pi-coding-agent does not export setThemeInstance/loadThemeFromPath.
  initTheme("dark", false);
  const markdownTheme = getMarkdownTheme();
  const selectTheme = getSelectListTheme();
  const editorTheme: EditorTheme = { borderColor: dim, selectList: selectTheme };

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

  const addAssistant = (message: AssistantMessage) => {
    transcript.addChild(
      new AssistantMessageComponent(displayMessage(message), false, markdownTheme),
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
      const panel = pairs.map(([key, value]) => `${dim(key.padEnd(width))}  ${value}`).join("\n");
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

  let current: AssistantMessageComponent | null = null;
  const toolComponents = new Map<string, ToolExecutionComponent>();
  agent.subscribe((event) => {
    if (event.type === "agent_start") {
      status.addChild(loader);
      loader.start();
    } else if (event.type === "agent_end") {
      loader.stop();
      status.clear();
      toolComponents.clear();
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
        current?.updateContent(displayMessage(event.message as AssistantMessage));
      }
    } else if (event.type === "message_end") {
      if (event.message.role === "assistant") {
        const message = event.message as AssistantMessage;
        if (message.stopReason !== "error" && message.stopReason !== "aborted") {
          applyCitationTrailer(message);
        }
        if (current) {
          // Replace the streaming component with the full final rendering
          // (markers and sources included).
          transcript.removeChild(current);
          current = null;
        }
        addAssistant(message);
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
      transcript.addChild(new Text(`\n${cyan("›")} ${trimmed}`));
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
