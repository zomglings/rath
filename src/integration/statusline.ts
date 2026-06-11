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
  assert.equal(even.cacheRead, 5, "25% of a 20-cell bar is 5 cells");
  assert.equal(even.input, 5, "25% of a 20-cell bar is 5 cells");
  assert.equal(even.cacheWrite + even.output, 0, "zero categories get no cells");
  assert.equal(even.empty, 10, "remaining cells are empty");

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
  log("Case 1 OK: bar apportionment (proportional, floored, clamped)");

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
