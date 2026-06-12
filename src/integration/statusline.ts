/**
 * Integration test: the TUI statusline. No API, no key.
 *
 * Covers the pure pieces (bar apportionment, token formatting, timestamp
 * formatting, full-line rendering with and without usage) and the one
 * subprocess boundary: gitInfo against throwaway repositories in each
 * working-tree state (clean, dirty, detached HEAD, not-a-repo).
 *
 * Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  barSegments,
  formatTimestamp,
  formatTokens,
  gitInfo,
  renderStatusline,
  STATUSLINE_BAR_WIDTH,
} from "../statusline.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

/** Strip ANSI escapes so assertions read the visible text. */
function plain(text: string): string {
  // biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes are control chars by definition.
  return text.replace(/\x1b\[[0-9;]*m/g, "");
}

function git(cwd: string, ...args: string[]): void {
  execFileSync("git", ["-C", cwd, ...args], { stdio: "ignore" });
}

async function main(): Promise<void> {
  // Case 1: bar apportionment — proportionality, the non-zero floor, clamping.
  const even = barSegments({ cacheRead: 50_000, cacheWrite: 0, input: 50_000, output: 0 }, 200_000);
  // 25% + 25% of a 30-cell bar: boundaries at round(7.5)=8 and round(15)=15,
  // so 8 + 7 cells — cumulative rounding keeps the total exact even when the
  // individual shares are not integral.
  assert.equal(even.cacheRead + even.input, STATUSLINE_BAR_WIDTH / 2, "50% fills half the bar");
  assert.ok(Math.abs(even.cacheRead - even.input) <= 1, "equal shares differ by at most a cell");
  assert.equal(even.cacheWrite + even.output, 0, "zero categories get no cells");
  assert.equal(even.empty, STATUSLINE_BAR_WIDTH / 2, "remaining cells are empty");

  const tiny = barSegments({ cacheRead: 0, cacheWrite: 0, input: 1, output: 1 }, 200_000);
  assert.equal(tiny.input, 1, "non-zero usage gets at least one cell");
  assert.equal(tiny.output, 1, "non-zero usage gets at least one cell");
  assert.equal(tiny.empty, STATUSLINE_BAR_WIDTH - 2, "floors come out of empty");

  const zeroWindow = barSegments({ cacheRead: 5, cacheWrite: 5, input: 5, output: 5 }, 0);
  assert.equal(
    zeroWindow.empty,
    STATUSLINE_BAR_WIDTH,
    "a zero window renders an empty bar, not a crash",
  );

  // Cumulative rounding: total fill tracks total usage — fill equals the
  // rounded cumulative boundary, not sum(floor(parts)). With a 30-cell bar
  // over 200k, one cell is ~6.67k tokens: 20k+20k is exactly 6 cells, and
  // growth to 24k+24k (7.2 cells) must move the boundary to 7 even though
  // each category alone only grew by 0.6 of a cell. (Values stay above one
  // cell each so the non-zero visibility floor does not kick in.)
  const a = barSegments({ cacheRead: 20_000, cacheWrite: 0, input: 20_000, output: 0 }, 200_000);
  assert.equal(
    a.cacheRead + a.input,
    Math.round((40_000 * STATUSLINE_BAR_WIDTH) / 200_000),
    "fill equals the rounded cumulative boundary",
  );
  const b = barSegments({ cacheRead: 24_000, cacheWrite: 0, input: 24_000, output: 0 }, 200_000);
  assert.equal(
    b.cacheRead + b.input,
    Math.round((48_000 * STATUSLINE_BAR_WIDTH) / 200_000),
    "growth moves the cumulative boundary",
  );
  assert.ok(b.cacheRead + b.input > a.cacheRead + a.input, "more usage fills more of the bar");

  // The floor still applies: sub-cell non-zero categories are visible.
  const floored = barSegments(
    { cacheRead: 4_000, cacheWrite: 0, input: 4_000, output: 0 },
    200_000,
  );
  assert.ok(floored.cacheRead >= 1 && floored.input >= 1, "sub-cell categories stay visible");
  log("Case 1 OK: bar apportionment (proportional, cumulative, floored, clamped)");

  // Case 2: token and timestamp formatting.
  assert.equal(formatTokens(999), "999");
  assert.equal(formatTokens(1000), "1k");
  assert.equal(formatTokens(123_456), "123k");
  const ts = formatTimestamp(new Date(2026, 1, 3, 4, 5, 6).getTime());
  assert.equal(ts, "2026-02-03T04:05:06", "local ISO 8601, zero-padded");
  log("Case 2 OK: token and timestamp formatting");

  // Case 3: full-line rendering, with and without usage.
  const line = plain(
    renderStatusline({
      model: "openai-native/gpt-5.5",
      contextWindow: 200_000,
      usage: { input: 10_000, output: 2_000, cacheRead: 88_000, cacheWrite: 0 },
      cwd: "/work/repo",
      git: { ref: "main", state: "clean" },
      lastInteraction: new Date(2026, 0, 1, 12, 0, 0).getTime(),
    }),
  );
  assert.ok(line.includes("openai-native/gpt-5.5"), "model is shown");
  assert.ok(line.includes("100k/200k (50%)"), `usage gauge is shown: ${line}`);
  assert.ok(line.includes("/work/repo"), "cwd is shown");
  assert.ok(line.includes("[main]"), "git ref is shown");
  assert.ok(line.includes("<2026-01-01T12:00:00>"), "last interaction is shown");

  const fresh = plain(
    renderStatusline({ model: "m/x", contextWindow: 200_000, cwd: "/work/repo" }),
  );
  assert.ok(fresh.includes("--"), "no usage renders an empty gauge");
  assert.ok(!fresh.includes("<"), "no last interaction, no timestamp");
  log("Case 3 OK: full-line rendering");

  // Case 4: gitInfo across repository states.
  const dir = mkdtempSync(join(tmpdir(), "rath-statusline-"));
  try {
    const repo = join(dir, "repo");
    execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
    git(repo, "config", "user.email", "test@example.com");
    git(repo, "config", "user.name", "Test");

    // An empty repo has no commits, but the unborn branch still has a name.
    assert.deepEqual(
      gitInfo(repo),
      { ref: "main", state: "clean" },
      "empty repository reports the unborn branch",
    );

    writeFileSync(join(repo, "file.txt"), "hello\n");
    git(repo, "add", "file.txt");
    git(repo, "commit", "-m", "initial");
    assert.deepEqual(gitInfo(repo), { ref: "main", state: "clean" }, "clean tree on a branch");

    writeFileSync(join(repo, "file.txt"), "changed\n");
    assert.deepEqual(gitInfo(repo), { ref: "main", state: "dirty" }, "dirty tree");
    git(repo, "checkout", "--", "file.txt");

    git(repo, "checkout", "--detach", "HEAD");
    const detached = gitInfo(repo);
    assert.ok(detached, "detached HEAD still reports");
    assert.match(detached.ref, /^[0-9a-f]{4,}$/, "detached HEAD falls back to a short hash");
    assert.equal(detached.state, "clean");

    assert.equal(gitInfo(dir), undefined, "not a repository -> undefined");
    log("Case 4 OK: gitInfo (unborn, clean, dirty, detached, not-a-repo)");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  log("statusline: all cases passed");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error);
    process.exit(1);
  },
);
