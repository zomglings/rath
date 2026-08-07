/** Public CLI integration for explicit Barbarian cleanup. Calls no API. */
import assert from "node:assert/strict";
import { execFileSync, spawn, spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { hostname, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { acquireBarbarianRunLock, createBarbarianArtifactRoot } from "../barbarian-artifacts.js";

const cli = join(dirname(dirname(fileURLToPath(import.meta.url))), "cli.js");

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" }).trim();
}

function createRepo(path: string): void {
  execFileSync("git", ["init", "-b", "main", path], { stdio: "ignore" });
  git(path, "config", "user.email", "test@example.com");
  git(path, "config", "user.name", "Test");
  writeFileSync(join(path, "tracked.txt"), "tracked\n");
  git(path, "add", "tracked.txt");
  git(path, "commit", "-m", "initial");
}

function addRun(repo: string, base: string): { artifactRoot: string; worktree: string } {
  const artifactRoot = createBarbarianArtifactRoot("solo", repo, base);
  const worktree = join(artifactRoot, "reproduction");
  git(repo, "worktree", "add", "--detach", worktree, "HEAD");
  return { artifactRoot, worktree };
}

function addLegacyRun(repo: string): { artifactRoot: string; worktree: string } {
  const artifactRoot = mkdtempSync(join(tmpdir(), "rath-barbarian-"));
  const worktree = join(artifactRoot, "reproduction");
  writeFileSync(
    join(artifactRoot, "checkpoint.json"),
    `${JSON.stringify({
      version: 1,
      repo,
      artifactRoot,
      source: "HEAD",
      target: "HEAD",
      modelSpec: "test/model",
      reasoning: "off",
      messages: [],
    })}\n`,
  );
  git(repo, "worktree", "add", "--detach", worktree, "HEAD");
  return { artifactRoot, worktree };
}

function run(
  args: string[],
  barbarianDirectory?: string,
): { status: number | null; stdout: string; stderr: string } {
  const env = { ...process.env };
  if (barbarianDirectory) {
    env.RATH_BARBARIAN_DIR = barbarianDirectory;
  } else {
    delete env.RATH_BARBARIAN_DIR;
  }
  const result = spawnSync(process.execPath, [cli, ...args], { encoding: "utf8", env });
  if (result.error) throw result.error;
  return { status: result.status, stdout: result.stdout, stderr: result.stderr };
}

function removeRegisteredWorktrees(repo: string): void {
  const output = git(repo, "worktree", "list", "--porcelain");
  for (const block of output.split(/\n\n+/)) {
    const path = /^worktree (.*)$/m.exec(block)?.[1];
    if (path && path !== repo && existsSync(path)) {
      execFileSync("git", ["-C", repo, "worktree", "remove", "--force", path], {
        stdio: "ignore",
      });
    }
  }
  execFileSync("git", ["-C", repo, "worktree", "prune", "--expire", "now"], {
    stdio: "ignore",
  });
}

async function waitForPath(path: string, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!existsSync(path)) {
    if (Date.now() >= deadline) throw new Error(`timed out waiting for ${path}`);
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 5));
  }
}

async function verifyStaleLockRace(group: string): Promise<void> {
  const root = join(group, "lock-race");
  const lock = join(root, ".review-lock");
  const ready = join(root, "ready");
  const permit = join(root, "permit");
  const release = join(root, "release");
  const worker = join(group, "lock-worker.mjs");
  mkdirSync(lock, { recursive: true });
  writeFileSync(
    join(lock, "owner.json"),
    JSON.stringify({
      version: 1,
      hostname: hostname(),
      pid: 2_147_483_647,
      token: "dead",
      startedAt: "2000-01-01T00:00:00.000Z",
    }),
  );
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, writeFileSync } from "node:fs";
const [role, root, ready, permit, release, artifactsModule] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalRename = fs.renameSync;
if (role === "B") {
  let stopped = false;
  let blockedRestore = false;
  fs.renameSync = (from, to, ...args) => {
    if (!stopped && from === root + "/.review-lock" && String(to).includes(".review-lock.stale-")) {
      stopped = true;
      writeFileSync(ready, "ready\\n");
      while (!existsSync(permit)) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5);
    }
    if (stopped && !blockedRestore && String(from).includes(".review-lock.stale-") && to === root + "/.review-lock") {
      blockedRestore = true;
      const error = new Error("simulated macOS destination contention");
      error.code = "ENOTEMPTY";
      throw error;
    }
    return originalRename(from, to, ...args);
  };
  syncBuiltinESMExports();
}
const { acquireBarbarianRunLock } = await import(artifactsModule);
try {
  const unlock = acquireBarbarianRunLock(root);
  writeFileSync(root + "/" + role + ".acquired", String(process.pid));
  while (!existsSync(release)) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5);
  unlock();
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
`,
  );
  const launch = (role: string) => {
    const child = spawn(
      process.execPath,
      [worker, role, root, ready, permit, release, artifactsModule],
      { stdio: ["ignore", "pipe", "pipe"] },
    );
    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk;
    });
    child.stderr.on("data", (chunk) => {
      output += chunk;
    });
    const exited = new Promise<number | null>((resolveExit, reject) => {
      child.on("error", reject);
      child.on("close", resolveExit);
    });
    return { child, exited, output: () => output };
  };
  const contender = launch("B");
  await Promise.race([
    waitForPath(ready),
    contender.exited.then((status) => {
      throw new Error(
        `lock contender exited before interleaving (${status}): ${contender.output()}`,
      );
    }),
  ]);
  const owner = launch("A");
  await waitForPath(join(root, "A.acquired"));
  writeFileSync(permit, "continue\n");
  const contenderStatus = await contender.exited;
  assert.equal(contenderStatus, 1, contender.output());
  assert.equal(
    existsSync(join(root, "B.acquired")),
    false,
    "stale takeover cannot displace a replacement live lock",
  );
  writeFileSync(release, "release\n");
  assert.equal(await owner.exited, 0, owner.output());
}

function verifyLockInitializationReplacement(group: string): void {
  const worker = join(group, "lock-initialization-replacement-worker.mjs");
  const root = join(group, "lock-initialization-replacement");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, mkdirSync, renameSync } from "node:fs";
import { hostname } from "node:os";
import { join } from "node:path";
const [root, artifactsModule] = process.argv.slice(2);
mkdirSync(root, { recursive: true });
const lock = join(root, ".review-lock");
const displaced = join(root, ".review-lock.original");
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalWrite = fs.writeFileSync;
let swapped = false;
fs.writeFileSync = (path, data, ...args) => {
  if (!swapped && path === join(lock, "owner.json")) {
    swapped = true;
    renameSync(lock, displaced);
    mkdirSync(lock);
    originalWrite(join(lock, "owner.json"), JSON.stringify({
      version: 1,
      hostname: hostname(),
      pid: process.pid,
      token: "replacement-owner",
      startedAt: new Date().toISOString(),
    }));
    const error = new Error("simulated owner write failure");
    error.code = "EIO";
    throw error;
  }
  return originalWrite(path, data, ...args);
};
syncBuiltinESMExports();
const { acquireBarbarianRunLock } = await import(artifactsModule);
assert.throws(() => acquireBarbarianRunLock(root), /simulated owner write failure/);
assert.equal(swapped, true);
assert.equal(existsSync(join(lock, "owner.json")), true, "replacement live lock is preserved");
assert.equal(existsSync(displaced), true, "displaced initializing lock is preserved");
`,
  );
  const result = spawnSync(process.execPath, [worker, root, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyCreationSymlinkRace(group: string): void {
  const worker = join(group, "creation-symlink-worker.mjs");
  const fixture = join(group, "creation-symlink-race");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { mkdirSync, readdirSync, symlinkSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const outside = join(fixture, "outside");
const base = join(fixture, "base");
mkdirSync(repo, { recursive: true });
mkdirSync(outside, { recursive: true });
symlinkSync(outside, base, "dir");
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalExists = fs.existsSync;
const originalMkdir = fs.mkdirSync;
let concealed = false;
let swapped = false;
fs.existsSync = (path) => {
  if (!concealed && path === base) {
    concealed = true;
    return false;
  }
  return originalExists(path);
};
fs.mkdirSync = (path, ...args) => {
  if (!swapped && path === base) {
    swapped = true;
    fs.unlinkSync(base);
    fs.symlinkSync(repo, base, "dir");
  }
  return originalMkdir(path, ...args);
};
syncBuiltinESMExports();
const { createBarbarianArtifactRoot } = await import(artifactsModule);
assert.throws(
  () => createBarbarianArtifactRoot("solo", repo, base),
  /outside the reviewed repository|changed during run creation/,
);
assert.equal(readdirSync(repo).some((entry) => entry.startsWith("solo-")), false);
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyCreationParentSwap(group: string): void {
  const worker = join(group, "creation-parent-swap-worker.mjs");
  const fixture = join(group, "creation-parent-swap");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, mkdirSync, readdirSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const base = join(fixture, "base");
const movedBase = join(fixture, "base-original");
mkdirSync(repo, { recursive: true });
mkdirSync(base, { recursive: true });
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalMkdtemp = fs.mkdtempSync;
let sentinel;
fs.mkdtempSync = (prefix, ...args) => {
  const created = originalMkdtemp(prefix, ...args);
  if (!sentinel) {
    renameSync(base, movedBase);
    mkdirSync(base);
    sentinel = join(base, "sentinel.txt");
    writeFileSync(sentinel, "preserve\\n");
  }
  return created;
};
syncBuiltinESMExports();
const { createBarbarianArtifactRoot } = await import(artifactsModule);
assert.throws(
  () => createBarbarianArtifactRoot("solo", repo, base),
  /barbarian directory changed during run creation/,
);
assert.equal(existsSync(sentinel), true, "failed validation preserves replacement content");
assert.equal(readdirSync(movedBase).some((entry) => entry.startsWith("solo-")), false);
assert.equal(readdirSync(fixture).some((entry) => entry.startsWith(".rath-barbarian-staging-")), false);
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyWritableBaseWithReadonlyParent(group: string): void {
  const worker = join(group, "readonly-parent-worker.mjs");
  const fixture = join(group, "readonly-parent");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const parent = join(fixture, "readonly-parent");
const base = join(parent, "runs");
execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
execFileSync("git", ["-C", repo, "config", "user.email", "test@example.com"]);
execFileSync("git", ["-C", repo, "config", "user.name", "Test"]);
writeFileSync(join(repo, "tracked.txt"), "tracked\\n");
execFileSync("git", ["-C", repo, "add", "tracked.txt"]);
execFileSync("git", ["-C", repo, "commit", "-m", "initial"]);
mkdirSync(base, { recursive: true });
chmodSync(parent, 0o555);
try {
  const { createBarbarianArtifactRoot } = await import(artifactsModule);
  const root = createBarbarianArtifactRoot("solo", repo, base);
  assert.equal(existsSync(root), true, "a writable configured base is sufficient");
} finally {
  chmodSync(parent, 0o755);
}
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyLockedPublication(group: string): void {
  const worker = join(group, "locked-publication-worker.mjs");
  const fixture = join(group, "locked-publication");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const base = join(fixture, "runs");
execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
execFileSync("git", ["-C", repo, "config", "user.email", "test@example.com"]);
execFileSync("git", ["-C", repo, "config", "user.name", "Test"]);
writeFileSync(join(repo, "tracked.txt"), "tracked\\n");
execFileSync("git", ["-C", repo, "add", "tracked.txt"]);
execFileSync("git", ["-C", repo, "commit", "-m", "initial"]);
mkdirSync(base);
const artifacts = await import(artifactsModule);
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalRename = fs.renameSync;
let concurrentClean;
fs.renameSync = (from, to, ...args) => {
  const result = originalRename(from, to, ...args);
  if (!concurrentClean && String(from).includes(".rath-barbarian-staging-") && String(to).includes("solo-")) {
    concurrentClean = artifacts.cleanBarbarianRuns({ repo, barbarianDir: base });
  }
  return result;
};
syncBuiltinESMExports();
const created = artifacts.createLockedBarbarianArtifactRoot("solo", repo, base);
assert.ok(concurrentClean);
assert.deepEqual(concurrentClean.removed, []);
assert.deepEqual(concurrentClean.skippedActive, [created.artifactRoot]);
assert.equal(existsSync(created.artifactRoot), true);
created.release();
const cleaned = artifacts.cleanBarbarianRuns({ repo, barbarianDir: base });
assert.equal(cleaned.removed.length, 1);
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyCleanupQuarantineRace(group: string): void {
  const worker = join(group, "cleanup-quarantine-worker.mjs");
  const fixture = join(group, "cleanup-quarantine-race");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const base = join(fixture, "runs");
const git = (...args) => execFileSync("git", ["-C", repo, ...args], { encoding: "utf8" }).trim();
execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
git("config", "user.email", "test@example.com");
git("config", "user.name", "Test");
writeFileSync(join(repo, "tracked.txt"), "tracked\\n");
git("add", "tracked.txt");
git("commit", "-m", "initial");
mkdirSync(base);
const { createBarbarianArtifactRoot } = await import(artifactsModule);
const root = createBarbarianArtifactRoot("solo", repo, base);
const worktree = join(root, "reproduction");
git("worktree", "add", "--detach", worktree, "HEAD");
const foreign = join(root, "foreign");
const sentinel = join(foreign, "sentinel.txt");
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalRename = fs.renameSync;
let inserted = false;
fs.renameSync = (from, to, ...args) => {
  const result = originalRename(from, to, ...args);
  if (!inserted && from === root && String(to).includes(".cleaning-")) {
    mkdirSync(join(to, "foreign"));
    writeFileSync(join(to, "foreign", "sentinel.txt"), "preserve\\n");
    inserted = true;
  }
  return result;
};
syncBuiltinESMExports();
const { cleanBarbarianRuns } = await import(artifactsModule);
const result = cleanBarbarianRuns({ repo, barbarianDir: base });
assert.equal(inserted, true, "fixture inserted content after quarantine");
assert.equal(result.removed.length, 0);
assert.match(result.errors.join("\\n"), /run content changed during quarantine/);
assert.equal(existsSync(sentinel), true, "atomic quarantine restores and preserves new content");
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyCleanupPublicPathRace(group: string): void {
  const worker = join(group, "cleanup-public-path-worker.mjs");
  const fixture = join(group, "cleanup-public-path-race");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const base = join(fixture, "runs");
const git = (...args) => execFileSync("git", ["-C", repo, ...args], { encoding: "utf8" }).trim();
execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
git("config", "user.email", "test@example.com");
git("config", "user.name", "Test");
writeFileSync(join(repo, "tracked.txt"), "tracked\\n");
git("add", "tracked.txt");
git("commit", "-m", "initial");
mkdirSync(base);
const { createBarbarianArtifactRoot } = await import(artifactsModule);
const root = createBarbarianArtifactRoot("solo", repo, base);
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalRename = fs.renameSync;
let inserted = false;
fs.renameSync = (from, to, ...args) => {
  const result = originalRename(from, to, ...args);
  if (!inserted && from === root && String(to).includes(".cleaning-")) {
    inserted = true;
    mkdirSync(root);
    writeFileSync(join(root, "sentinel.txt"), "preserve\\n");
  }
  return result;
};
syncBuiltinESMExports();
const { cleanBarbarianRuns } = await import(artifactsModule);
const result = cleanBarbarianRuns({ repo, barbarianDir: base });
assert.equal(inserted, true);
assert.equal(result.removed.length, 0);
assert.match(result.errors.join("\\n"), /public path reappeared/);
assert.equal(existsSync(join(root, "sentinel.txt")), true);
assert.equal(readdirSync(base).some((entry) => entry.includes(".cleaning-")), true);
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyCleanupFinalRemovalRace(group: string): void {
  const worker = join(group, "cleanup-final-removal-worker.mjs");
  const fixture = join(group, "cleanup-final-removal-race");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createRequire, syncBuiltinESMExports } from "node:module";
import { existsSync, mkdirSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const base = join(fixture, "runs");
execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
execFileSync("git", ["-C", repo, "config", "user.email", "test@example.com"]);
execFileSync("git", ["-C", repo, "config", "user.name", "Test"]);
writeFileSync(join(repo, "tracked.txt"), "tracked\\n");
execFileSync("git", ["-C", repo, "add", "tracked.txt"]);
execFileSync("git", ["-C", repo, "commit", "-m", "initial"]);
mkdirSync(base);
const artifacts = await import(artifactsModule);
const root = artifacts.createBarbarianArtifactRoot("solo", repo, base);
const require = createRequire(import.meta.url);
const fs = require("node:fs");
const originalRmdir = fs.rmdirSync;
let replacement;
let original;
fs.rmdirSync = (path, ...args) => {
  if (!replacement && dirname(path) === base && String(path).includes(".cleaning-")) {
    original = path + ".original";
    renameSync(path, original);
    mkdirSync(path);
    replacement = join(path, "ordinary-user-data.txt");
    writeFileSync(replacement, "preserve\\n");
  }
  return originalRmdir(path, ...args);
};
syncBuiltinESMExports();
const result = artifacts.cleanBarbarianRuns({ repo, barbarianDir: base });
assert.equal(result.removed.length, 0);
assert.equal(existsSync(replacement), true, "replacement data survives final removal race");
assert.equal(existsSync(original), true, "the emptied original run remains separately identifiable");
assert.match(result.errors.join("\\n"), /directory not empty|ENOTEMPTY/i);
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

function verifyDryRunWorktreeContainment(group: string): void {
  const worker = join(group, "dry-run-worktree-containment-worker.mjs");
  const fixture = join(group, "dry-run-worktree-containment");
  const artifactsModule = new URL("../barbarian-artifacts.js", import.meta.url).href;
  writeFileSync(
    worker,
    `import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, renameSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const [fixture, artifactsModule] = process.argv.slice(2);
const repo = join(fixture, "repo");
const base = join(fixture, "runs");
const outside = join(fixture, "outside");
const git = (...args) => execFileSync("git", ["-C", repo, ...args], { encoding: "utf8" }).trim();
execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
git("config", "user.email", "test@example.com");
git("config", "user.name", "Test");
writeFileSync(join(repo, "tracked.txt"), "tracked\\n");
git("add", "tracked.txt");
git("commit", "-m", "initial");
mkdirSync(base);
mkdirSync(outside);
const artifacts = await import(artifactsModule);
const root = artifacts.createBarbarianArtifactRoot("solo", repo, base);
const worktree = join(root, "reproduction");
const moved = join(outside, "reproduction");
git("worktree", "add", "--detach", worktree, "HEAD");
renameSync(worktree, moved);
writeFileSync(join(moved, "sentinel.txt"), "preserve\\n");
symlinkSync(moved, worktree, "dir");
const dryRun = artifacts.cleanBarbarianRuns({ repo, barbarianDir: base, dryRun: true });
const real = artifacts.cleanBarbarianRuns({ repo, barbarianDir: base });
assert.equal(dryRun.wouldRemove.length, 0);
assert.equal(real.removed.length, 0);
assert.match(dryRun.errors.join("\\n"), /worktree resolves outside its run root/);
assert.match(real.errors.join("\\n"), /worktree resolves outside its run root/);
assert.equal(existsSync(join(moved, "sentinel.txt")), true);
`,
  );
  const result = spawnSync(process.execPath, [worker, fixture, artifactsModule], {
    encoding: "utf8",
  });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, result.stderr || result.stdout);
}

async function main(): Promise<void> {
  const group = realpathSync(mkdtempSync(join(tmpdir(), "rath-barbarian-clean-test-")));
  const repo = join(group, "repo");
  const foreignRepo = join(group, "foreign-repo");
  const base = join(group, "barbarian");
  const outsideBase = join(group, "outside");
  mkdirSync(base);
  mkdirSync(outsideBase);
  createRepo(repo);
  createRepo(foreignRepo);

  let legacyRoot: string | undefined;
  let invalidLegacyRoot: string | undefined;
  let invalidLegacyHordeRoot: string | undefined;
  try {
    const cleanable = addRun(repo, base);
    const foreign = addRun(foreignRepo, base);
    const outside = addRun(repo, outsideBase);
    const active = addRun(repo, base);
    const releaseActive = acquireBarbarianRunLock(active.artifactRoot);

    const unverifiedRoot = mkdtempSync(join(base, "solo-"));
    const unverifiedWorktree = join(unverifiedRoot, "reproduction");
    git(repo, "worktree", "add", "--detach", unverifiedWorktree, "HEAD");

    const corruptNewRoot = join(base, "solo-ABC123");
    mkdirSync(corruptNewRoot);
    writeFileSync(join(corruptNewRoot, "checkpoint.json"), JSON.stringify({ repo }));
    writeFileSync(join(corruptNewRoot, "sentinel.txt"), "preserve\n");

    invalidLegacyRoot = mkdtempSync(join(tmpdir(), "rath-barbarian-"));
    writeFileSync(
      join(invalidLegacyRoot, "checkpoint.json"),
      JSON.stringify({ version: 1, repo, artifactRoot: invalidLegacyRoot }),
    );
    writeFileSync(join(invalidLegacyRoot, "sentinel.txt"), "preserve\n");

    invalidLegacyHordeRoot = mkdtempSync(join(tmpdir(), "rath-barbarian-horde-"));
    writeFileSync(
      join(invalidLegacyHordeRoot, "checkpoint.json"),
      JSON.stringify({
        version: 2,
        mode: "horde",
        repo,
        artifactRoot: invalidLegacyHordeRoot,
      }),
    );
    writeFileSync(join(invalidLegacyHordeRoot, "sentinel.txt"), "preserve\n");

    const statusBeforeRejectedDirectory = git(repo, "status", "--porcelain=v1", "-uall");
    assert.throws(
      () => createBarbarianArtifactRoot("solo", repo, join(repo, ".barbarian-runs")),
      /must be outside the reviewed repository/,
    );
    assert.equal(
      git(repo, "status", "--porcelain=v1", "-uall"),
      statusBeforeRejectedDirectory,
      "rejecting an in-repository Barbarian directory creates no files",
    );
    assert.equal(
      existsSync(join(repo, ".barbarian-runs")),
      false,
      "rejecting a nonexistent in-repository directory leaves no empty directory",
    );

    const inRepoBase = join(repo, ".barbarian-clean-test");
    const inRepoRoot = join(inRepoBase, "solo-ABC123");
    mkdirSync(inRepoRoot, { recursive: true });
    writeFileSync(
      join(inRepoRoot, "run.json"),
      JSON.stringify({
        version: 1,
        mode: "solo",
        repo,
        artifactRoot: inRepoRoot,
        createdAt: "2026-01-01T00:00:00.000Z",
      }),
    );
    writeFileSync(join(inRepoRoot, "sentinel.txt"), "preserve\n");
    const rejectedInRepoClean = run([
      "barbarian",
      "clean",
      "--repo",
      repo,
      "--barbarian-dir",
      inRepoBase,
    ]);
    assert.equal(rejectedInRepoClean.status, 1);
    assert.match(rejectedInRepoClean.stderr, /must be outside the reviewed repository/);
    assert.ok(
      existsSync(join(inRepoRoot, "sentinel.txt")),
      "cleanup rejects an in-repository Barbarian directory",
    );

    const dryRun = run(["barbarian", "clean", "--repo", repo, "--dry-run"], base);
    assert.equal(dryRun.status, 0, dryRun.stderr);
    assert.match(dryRun.stdout, new RegExp(`Would remove ${cleanable.artifactRoot}`));
    assert.match(dryRun.stdout, new RegExp(`Skipped active run ${active.artifactRoot}`));
    assert.doesNotMatch(dryRun.stdout, new RegExp(foreign.artifactRoot));
    assert.doesNotMatch(dryRun.stdout, new RegExp(outside.artifactRoot));
    assert.doesNotMatch(dryRun.stdout, new RegExp(unverifiedRoot));
    assert.ok(existsSync(cleanable.worktree), "dry-run retains the selected worktree");

    const cleaned = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(cleaned.status, 0, cleaned.stderr);
    assert.match(cleaned.stdout, new RegExp(`Removed ${cleanable.artifactRoot}`));
    assert.ok(!existsSync(cleanable.artifactRoot), "clean removes the verified run root");
    assert.ok(existsSync(active.worktree), "clean skips a live locked run");
    assert.ok(existsSync(foreign.worktree), "clean does not cross repository ownership");
    assert.ok(existsSync(outside.worktree), "clean does not cross directory containment");
    assert.ok(existsSync(unverifiedWorktree), "clean does not remove unverified directories");
    assert.ok(
      existsSync(join(corruptNewRoot, "sentinel.txt")),
      "clean rejects checkpoint-only ownership for new roots",
    );
    assert.ok(
      existsSync(join(invalidLegacyRoot, "sentinel.txt")),
      "clean rejects malformed legacy checkpoints",
    );
    assert.ok(
      existsSync(join(invalidLegacyHordeRoot, "sentinel.txt")),
      "clean rejects incomplete legacy horde checkpoints",
    );

    releaseActive();
    const cleanedFormerlyActive = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(cleanedFormerlyActive.status, 0, cleanedFormerlyActive.stderr);
    assert.ok(!existsSync(active.artifactRoot), "an unlocked run becomes cleanable");

    const displaced = addRun(repo, base);
    const releaseDisplaced = acquireBarbarianRunLock(displaced.artifactRoot);
    const displacedLock = join(displaced.artifactRoot, ".review-lock.stale-test");
    renameSync(join(displaced.artifactRoot, ".review-lock"), displacedLock);
    const displacedDryRun = run(["barbarian", "clean", "--repo", repo, "--dry-run"], base);
    assert.equal(displacedDryRun.status, 0, displacedDryRun.stderr);
    assert.match(
      displacedDryRun.stdout,
      new RegExp(`Skipped active run ${displaced.artifactRoot}`),
    );
    releaseDisplaced();
    assert.equal(
      existsSync(displacedLock),
      false,
      "release removes its token-owned displaced lock",
    );
    const cleanedDisplaced = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(cleanedDisplaced.status, 0, cleanedDisplaced.stderr);
    assert.ok(!existsSync(displaced.artifactRoot));

    const guarded = addRun(repo, base);
    const nestedForeignWorktree = join(guarded.artifactRoot, "foreign-worktree");
    git(foreignRepo, "worktree", "add", "--detach", nestedForeignWorktree, "HEAD");
    writeFileSync(join(nestedForeignWorktree, "sentinel.txt"), "preserve\n");
    const guardedClean = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(guardedClean.status, 1);
    assert.match(guardedClean.stderr, /unowned Git worktrees or repositories/);
    assert.ok(
      existsSync(join(nestedForeignWorktree, "sentinel.txt")),
      "cleanup preserves a foreign worktree nested beneath an owned run",
    );
    assert.match(
      git(foreignRepo, "worktree", "list", "--porcelain"),
      new RegExp(nestedForeignWorktree),
    );
    git(foreignRepo, "worktree", "remove", "--force", nestedForeignWorktree);
    const cleanedGuarded = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(cleanedGuarded.status, 0, cleanedGuarded.stderr);
    assert.ok(!existsSync(guarded.artifactRoot));

    const guardedBare = addRun(repo, base);
    const nestedBareRepo = join(guardedBare.artifactRoot, "foreign.git");
    execFileSync("git", ["init", "--bare", nestedBareRepo], { stdio: "ignore" });
    const guardedBareDryRun = run(["barbarian", "clean", "--repo", repo, "--dry-run"], base);
    assert.equal(guardedBareDryRun.status, 1);
    assert.doesNotMatch(guardedBareDryRun.stdout, new RegExp(guardedBare.artifactRoot));
    assert.match(guardedBareDryRun.stderr, /unowned Git worktrees or repositories/);
    const guardedBareClean = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(guardedBareClean.status, 1);
    assert.ok(existsSync(join(nestedBareRepo, "HEAD")), "cleanup preserves nested bare repos");
    rmSync(nestedBareRepo, { recursive: true, force: true });
    const cleanedGuardedBare = run(["barbarian", "clean", "--repo", repo], base);
    assert.equal(cleanedGuardedBare.status, 0, cleanedGuardedBare.stderr);
    assert.ok(!existsSync(guardedBare.artifactRoot));

    if (process.platform !== "win32") {
      await verifyStaleLockRace(group);
      verifyLockInitializationReplacement(group);
      verifyCreationSymlinkRace(group);
      verifyCreationParentSwap(group);
      verifyWritableBaseWithReadonlyParent(group);
      verifyLockedPublication(group);
      verifyCleanupQuarantineRace(group);
      verifyCleanupPublicPathRace(group);
      verifyCleanupFinalRemovalRace(group);
      verifyDryRunWorktreeContainment(group);
      const raced = addRun(repo, base);
      const racedWorktrees = [raced.worktree];
      for (let index = 1; index < 10; index++) {
        const worktree = join(raced.artifactRoot, `reproduction-${index}`);
        git(repo, "worktree", "add", "--detach", worktree, "HEAD");
        racedWorktrees.push(worktree);
      }
      const cleaner = spawn(
        process.execPath,
        [cli, "barbarian", "clean", "--repo", repo, "--barbarian-dir", base],
        { env: { ...process.env, RATH_BARBARIAN_DIR: base }, stdio: ["ignore", "pipe", "pipe"] },
      );
      let cleanerStdout = "";
      let cleanerStderr = "";
      cleaner.stdout.on("data", (chunk) => {
        cleanerStdout += chunk;
      });
      cleaner.stderr.on("data", (chunk) => {
        cleanerStderr += chunk;
      });
      const firstRemoval = [...racedWorktrees].sort(
        (left, right) => right.length - left.length,
      )[0]!;
      while (existsSync(firstRemoval) && cleaner.exitCode === null) {
        await new Promise((resolveDelay) => setTimeout(resolveDelay, 1));
      }
      assert.equal(
        cleaner.exitCode,
        null,
        "cleanup remains active after its first worktree removal",
      );
      cleaner.kill("SIGSTOP");
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 20));
      let competingLockError: unknown;
      try {
        acquireBarbarianRunLock(raced.artifactRoot);
      } catch (error) {
        competingLockError = error;
      }
      const movedRoot = join(group, "moved-run");
      renameSync(raced.artifactRoot, movedRoot);
      symlinkSync(movedRoot, raced.artifactRoot, "dir");
      writeFileSync(join(movedRoot, "sentinel.txt"), "outside\n");
      cleaner.kill("SIGCONT");
      assert.match(
        competingLockError instanceof Error ? competingLockError.message : "",
        /barbarian review is already active/,
        "cleanup holds the run lock throughout destructive work",
      );
      const cleanerStatus = await new Promise<number | null>((resolveStatus, reject) => {
        cleaner.on("error", reject);
        cleaner.on("close", resolveStatus);
      });
      assert.equal(cleanerStatus, 1, cleanerStdout);
      assert.match(
        cleanerStderr,
        /Barbarian run changed during cleanup|worktree resolves outside its run root/,
      );
      assert.ok(
        existsSync(join(movedRoot, "sentinel.txt")),
        "identity change preserves moved content",
      );
      assert.ok(
        racedWorktrees.slice(1).some((worktree) => existsSync(worktree)),
        "identity change stops subsequent worktree removals",
      );
    }

    const cleanedOutside = run(
      ["barbarian", "clean", "--repo", repo, "--barbarian-dir", outsideBase],
      base,
    );
    assert.equal(cleanedOutside.status, 0, cleanedOutside.stderr);
    assert.ok(!existsSync(outside.artifactRoot), "an explicit directory scopes cleanup");

    const cleanedForeign = run(["barbarian", "clean", "--repo", foreignRepo], base);
    assert.equal(cleanedForeign.status, 0, cleanedForeign.stderr);
    assert.ok(!existsSync(foreign.artifactRoot), "the owning repository can clean its run");

    const legacy = addLegacyRun(repo);
    legacyRoot = legacy.artifactRoot;
    const unrelatedPrunable = join(group, "unrelated-prunable");
    const movedUnrelated = join(group, "unrelated-prunable-moved");
    git(repo, "worktree", "add", "--detach", unrelatedPrunable, "HEAD");
    renameSync(unrelatedPrunable, movedUnrelated);
    rmSync(legacy.worktree, { recursive: true, force: true });
    assert.match(git(repo, "worktree", "list", "--porcelain"), /prunable/);
    const cleanedLegacy = run(["barbarian", "clean", "--repo", repo]);
    assert.equal(cleanedLegacy.status, 0, cleanedLegacy.stderr);
    assert.ok(!existsSync(legacy.artifactRoot), "default cleanup recognizes legacy run roots");
    assert.doesNotMatch(
      git(repo, "worktree", "list", "--porcelain"),
      new RegExp(legacy.worktree),
      "legacy prunable metadata is removed",
    );
    assert.match(
      git(repo, "worktree", "list", "--porcelain"),
      new RegExp(unrelatedPrunable),
      "targeted cleanup preserves unrelated prunable metadata",
    );

    assert.ok(
      readFileSync(join(unverifiedWorktree, ".git"), "utf8").includes("worktrees"),
      "negative control remains a registered Git worktree",
    );
    process.stdout.write(
      "Barbarian clean verified ownership schemas, containment, nested repositories, lock races, dry-run, and targeted legacy pruning.\n",
    );
  } finally {
    removeRegisteredWorktrees(repo);
    removeRegisteredWorktrees(foreignRepo);
    if (legacyRoot) rmSync(legacyRoot, { recursive: true, force: true });
    if (invalidLegacyRoot) rmSync(invalidLegacyRoot, { recursive: true, force: true });
    if (invalidLegacyHordeRoot) {
      rmSync(invalidLegacyHordeRoot, { recursive: true, force: true });
    }
    rmSync(group, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : error);
  process.exit(1);
});
