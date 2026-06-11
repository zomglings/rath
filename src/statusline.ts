/**
 * The TUI statusline: a single line below the editor summarizing the session
 * at a glance, in the style of a shell prompt or Claude Code's statusline:
 *
 *   <model>  [context bar] used/window (pct%) | <cwd> [git ref] <last turn>
 *
 * The context bar is a fixed-width gauge of the model's context window, with
 * one colored segment per token category from the last assistant message's
 * usage: cache reads (green), cache writes (yellow), fresh input (orange),
 * output (dim). Unused window is dark-gray shade. Any non-zero category gets
 * at least one cell so small-but-present usage is visible.
 *
 * The git ref is colored by working-tree state: red when a merge/rebase is in
 * progress, yellow when the tree is dirty, green when clean. The last-turn
 * timestamp is the wall-clock time the previous turn finished (ISO 8601,
 * pink) — a quick answer to "how stale is this transcript?".
 *
 * Everything here is pure computation plus one subprocess boundary (gitInfo);
 * rendering with real terminal colors keeps the module independent of pi-tui
 * so it can be unit-tested without a terminal.
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { isAbsolute, join } from "node:path";

/** Token counts from the last assistant turn, by context-window category. */
export interface ContextUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
}

export type GitTreeState = "conflict" | "dirty" | "clean";

export interface GitInfo {
  /** Branch name, or short commit hash when detached. */
  ref: string;
  state: GitTreeState;
}

export interface StatuslineData {
  /** Model spec as shown to the user (provider/model-id). */
  model: string;
  /** The model's context window, in tokens. */
  contextWindow: number;
  /** Usage of the last assistant message; undefined before the first turn. */
  usage?: ContextUsage;
  cwd: string;
  /** Git state of cwd; undefined outside a repository. */
  git?: GitInfo;
  /** Wall-clock time (epoch ms) the last turn finished. */
  lastInteraction?: number;
}

export const STATUSLINE_BAR_WIDTH = 20;

/** Cells of the context bar assigned to each usage category. */
export interface BarSegments {
  cacheRead: number;
  cacheWrite: number;
  input: number;
  output: number;
  empty: number;
}

/**
 * Apportion the bar's cells to the usage categories, proportional to the
 * context window, with a floor of one cell for any non-zero category (small
 * usage must be visible). The floors can push the total past the bar width
 * for tiny windows; empty clamps at zero and the bar renders slightly long
 * rather than lying by omission.
 */
export function barSegments(
  usage: ContextUsage,
  contextWindow: number,
  barWidth = STATUSLINE_BAR_WIDTH,
): BarSegments {
  const cell = (tokens: number): number => {
    if (tokens <= 0 || contextWindow <= 0) {
      return 0;
    }
    return Math.max(1, Math.floor((tokens * barWidth) / contextWindow));
  };
  const segments = {
    cacheRead: cell(usage.cacheRead),
    cacheWrite: cell(usage.cacheWrite),
    input: cell(usage.input),
    output: cell(usage.output),
  };
  const filled = segments.cacheRead + segments.cacheWrite + segments.input + segments.output;
  return { ...segments, empty: Math.max(0, barWidth - filled) };
}

/** 12345 -> "12k"; counts below 1000 are shown exactly. */
export function formatTokens(n: number): string {
  return n >= 1000 ? `${Math.floor(n / 1000)}k` : String(n);
}

/**
 * Git ref and working-tree state for `cwd`, or undefined when it is not in a
 * git repository (or git is not installed). Synchronous subprocess calls: the
 * caller refreshes on turn boundaries, not per render frame.
 */
export function gitInfo(cwd: string): GitInfo | undefined {
  const git = (...args: string[]): { ok: boolean; out: string } => {
    try {
      const result = spawnSync("git", ["-C", cwd, ...args], { encoding: "utf8" });
      return { ok: result.status === 0, out: (result.stdout ?? "").trim() };
    } catch {
      return { ok: false, out: "" };
    }
  };
  const gitDir = git("rev-parse", "--git-dir");
  if (!gitDir.ok) {
    return undefined;
  }
  const dir = isAbsolute(gitDir.out) ? gitDir.out : join(cwd, gitDir.out);
  let state: GitTreeState;
  if (
    existsSync(join(dir, "MERGE_HEAD")) ||
    existsSync(join(dir, "rebase-merge")) ||
    existsSync(join(dir, "rebase-apply"))
  ) {
    state = "conflict";
  } else {
    const status = git("status", "--porcelain");
    state = status.ok && status.out.length > 0 ? "dirty" : "clean";
  }
  const branch = git("symbolic-ref", "--short", "HEAD");
  if (branch.ok && branch.out.length > 0) {
    return { ref: branch.out, state };
  }
  const commit = git("rev-parse", "--short", "HEAD");
  // Detached HEAD has no branch name; fall back to the short hash. (An empty
  // repository's unborn branch is named, so it is handled by the branch path.)
  return commit.ok && commit.out.length > 0 ? { ref: commit.out, state } : undefined;
}

// Statusline palette. Plain ANSI codes (not pi-tui theme styles) so the module
// stays terminal-library-free; the statusline is a fixed informational strip,
// not themed content.
const RESET = "\x1b[0m";
const BOLD_CYAN = "\x1b[1;36m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const RED = "\x1b[31m";
const ORANGE = "\x1b[38;5;208m";
const DIM_WHITE = "\x1b[2;37m";
const DARK_GRAY = "\x1b[90m";
const BLUE = "\x1b[34m";
const PINK = "\x1b[38;2;255;182;193m";

const FULL_BLOCK = "█";
const LIGHT_SHADE = "░";

const GIT_STATE_COLOR: Record<GitTreeState, string> = {
  conflict: RED,
  dirty: YELLOW,
  clean: GREEN,
};

/** Local wall-clock ISO 8601 (no timezone suffix), e.g. 2026-02-11T09:30:12. */
export function formatTimestamp(epochMs: number): string {
  const d = new Date(epochMs);
  const pad = (n: number): string => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

/** Render the statusline as a single ANSI-colored line (no trailing newline). */
export function renderStatusline(data: StatuslineData): string {
  const model = `${BOLD_CYAN}${data.model}${RESET}`;

  let gauge: string;
  if (data.usage && data.contextWindow > 0) {
    const { input, output, cacheRead, cacheWrite } = data.usage;
    const used = input + output + cacheRead + cacheWrite;
    const seg = barSegments(data.usage, data.contextWindow);
    const bar =
      GREEN +
      FULL_BLOCK.repeat(seg.cacheRead) +
      YELLOW +
      FULL_BLOCK.repeat(seg.cacheWrite) +
      ORANGE +
      FULL_BLOCK.repeat(seg.input) +
      DIM_WHITE +
      FULL_BLOCK.repeat(seg.output) +
      DARK_GRAY +
      LIGHT_SHADE.repeat(seg.empty) +
      RESET;
    const pct = Math.floor((used * 100) / data.contextWindow);
    gauge = `[${bar}] ${formatTokens(used)}/${formatTokens(data.contextWindow)} (${pct}%)`;
  } else {
    gauge = `[${DARK_GRAY}${LIGHT_SHADE.repeat(STATUSLINE_BAR_WIDTH)}${RESET}] --`;
  }

  const cwd = `${BLUE}${data.cwd}${RESET}`;
  const git = data.git ? ` ${GIT_STATE_COLOR[data.git.state]}[${data.git.ref}]${RESET}` : "";
  const time = data.lastInteraction
    ? ` ${PINK}<${formatTimestamp(data.lastInteraction)}>${RESET}`
    : "";

  return `${model}  ${gauge} ${DARK_GRAY}|${RESET} ${cwd}${git}${time}`;
}
