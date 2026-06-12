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

export const STATUSLINE_BAR_WIDTH = 50;

/** Cells of the context bar assigned to each usage category. */
export interface BarSegments {
  cacheRead: number;
  cacheWrite: number;
  input: number;
  output: number;
  empty: number;
}

/**
 * Apportion the bar's cells to the usage categories. Boundaries are placed by
 * cumulative rounding — each category's cell count is the rounded cumulative
 * total minus the cells already assigned — so the bar's overall fill tracks
 * total usage (growth spread across categories still moves the boundary; the
 * per-category-floor() alternative swallows it). Any non-zero category still
 * gets at least one cell so small-but-present usage is visible; the floors
 * can push the total past the bar width for tiny windows, in which case empty
 * clamps at zero and the bar renders slightly long rather than lying by
 * omission.
 */
export function barSegments(
  usage: ContextUsage,
  contextWindow: number,
  barWidth = STATUSLINE_BAR_WIDTH,
): BarSegments {
  if (contextWindow <= 0) {
    return { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, empty: barWidth };
  }
  // Bar order: cache reads, cache writes, fresh input, output.
  const counts = [usage.cacheRead, usage.cacheWrite, usage.input, usage.output];
  const cells: number[] = [];
  let cumulative = 0;
  let assigned = 0;
  for (const tokens of counts) {
    cumulative += Math.max(0, tokens);
    let n = Math.round((cumulative * barWidth) / contextWindow) - assigned;
    if (tokens > 0 && n <= 0) {
      n = 1; // visibility floor: non-zero usage must show
    } else if (n < 0) {
      n = 0; // an earlier floor overshot the boundary; do not go negative
    }
    assigned += n;
    cells.push(n);
  }
  return {
    cacheRead: cells[0]!,
    cacheWrite: cells[1]!,
    input: cells[2]!,
    output: cells[3]!,
    empty: Math.max(0, barWidth - assigned),
  };
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
// not themed content. All colors are 256-color palette indices with no
// attributes (no bold/dim): attribute changes can make terminals substitute
// font weights, whose block glyphs may not align with the regular face — the
// bar must be one seamless strip. (Notably, dim (SGR 2) is sticky: a later
// color code does not clear it, and dim is exactly the attribute terminals
// most often render with a thinner font.)
const RESET = "\x1b[0m";
const BOLD_CYAN = "\x1b[1;36m";
const GREEN = "\x1b[38;5;2m";
const YELLOW = "\x1b[38;5;3m";
const RED = "\x1b[38;5;1m";
const ORANGE = "\x1b[38;5;208m";
const DIM_WHITE = "\x1b[38;5;245m";
const DARK_GRAY = "\x1b[38;5;238m";
const BLUE = "\x1b[38;5;4m";
// 218 is the closest xterm-256 index to light pink (255,182,193); truecolor
// (38;2;r;g;b) is avoided because macOS Terminal.app does not support it.
const PINK = "\x1b[38;5;218m";

const FULL_BLOCK = "█";
const LIGHT_SHADE = "░";
// Leading-edge partial cell, in eighths: PARTIAL_BLOCKS[k-1] is k/8 of a cell.
// One cell of a 50-wide bar over a 200k window is 4k tokens — still too coarse
// to see a single turn's growth — so the first empty cell renders as a partial
// block, giving ~8x finer visible progress (~500 tokens per visible step).
const PARTIAL_BLOCKS = ["▏", "▎", "▍", "▌", "▋", "▊", "▉"] as const;

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
    const filled = seg.cacheRead + seg.cacheWrite + seg.input + seg.output;
    // Sub-cell progress: when the exact fill exceeds the whole cells drawn,
    // render the leading empty cell as a partial block in the color of the
    // last non-zero category. Skipped when the visibility floors already
    // overdrew (exact < filled) or the bar is full.
    let partial = "";
    let empty = seg.empty;
    if (empty > 0) {
      const exact = (used * STATUSLINE_BAR_WIDTH) / data.contextWindow;
      const eighths = Math.floor((exact - filled) * 8);
      if (eighths >= 1) {
        const color = output > 0 ? DIM_WHITE : input > 0 ? ORANGE : cacheWrite > 0 ? YELLOW : GREEN;
        partial = `${color}${PARTIAL_BLOCKS[Math.min(eighths, 7) - 1]}`;
        empty -= 1;
      }
    }
    const bar =
      GREEN +
      FULL_BLOCK.repeat(seg.cacheRead) +
      YELLOW +
      FULL_BLOCK.repeat(seg.cacheWrite) +
      ORANGE +
      FULL_BLOCK.repeat(seg.input) +
      DIM_WHITE +
      FULL_BLOCK.repeat(seg.output) +
      partial +
      DARK_GRAY +
      LIGHT_SHADE.repeat(empty) +
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
