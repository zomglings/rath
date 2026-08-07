/**
 * Barbarian artifact ownership, locking, and explicit cleanup.
 *
 * Reviews retain their complete run directories. `rath barbarian clean` is the
 * only production path that removes them, after verifying both repository
 * ownership and containment beneath the configured Barbarian directory.
 */
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  type Dirent,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmdirSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { hostname, tmpdir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";

const RUN_FILE = "run.json";
const CHECKPOINT_FILE = "checkpoint.json";
const LOCK_DIR = ".review-lock";
const LOCK_INITIALIZATION_GRACE_MS = 10_000;
const LEGACY_RUN = /^rath-barbarian-(?:horde-)?[A-Za-z0-9]{6}$/;
const NEW_RUN = /^(?:solo|horde)-(?:[A-Za-z0-9]{6}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$/;

const GIT_REPOSITORY_ENV_KEYS = [
  "GIT_DIR",
  "GIT_WORK_TREE",
  "GIT_COMMON_DIR",
  "GIT_INDEX_FILE",
  "GIT_OBJECT_DIRECTORY",
  "GIT_ALTERNATE_OBJECT_DIRECTORIES",
  "GIT_CEILING_DIRECTORIES",
] as const;

export type BarbarianMode = "solo" | "horde";

interface BarbarianRun {
  version: 1;
  mode: BarbarianMode;
  repo: string;
  artifactRoot: string;
  createdAt: string;
}

interface ReviewLockOwner {
  version: 1;
  hostname: string;
  pid: number;
  token: string;
  startedAt: string;
}

interface WorktreeRecord {
  path: string;
  prunable: boolean;
}

interface RunIdentity {
  artifactRoot: string;
  canonicalRoot: string;
  device: number;
  inode: number;
}

interface FileIdentity {
  device: number;
  inode: number;
}

export class BarbarianRunActiveError extends Error {}

export type BarbarianRunLockRelease = (artifactRootOverride?: string) => void;

export interface BarbarianCleanOptions {
  repo: string;
  barbarianDir?: string;
  dryRun?: boolean;
}

export interface BarbarianCleanRun {
  artifactRoot: string;
  worktrees: string[];
}

export interface BarbarianCleanResult {
  removed: BarbarianCleanRun[];
  wouldRemove: BarbarianCleanRun[];
  skippedActive: string[];
  errors: string[];
}

function gitEnvironment(): NodeJS.ProcessEnv {
  const environment = { ...process.env };
  for (const key of GIT_REPOSITORY_ENV_KEYS) {
    delete environment[key];
  }
  return environment;
}

function errno(error: unknown): string | undefined {
  return error instanceof Error ? (error as NodeJS.ErrnoException).code : undefined;
}

function samePath(left: string, right: string): boolean {
  return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;
}

function canonicalExisting(path: string): string {
  return realpathSync(resolve(path));
}

/** Resolve an existing Barbarian run path through filesystem aliases. */
export function canonicalBarbarianArtifactRoot(path: string): string {
  return canonicalExisting(path);
}

function canonicalPotential(path: string): string {
  let cursor = resolve(path);
  const missing: string[] = [];
  while (!existsSync(cursor)) {
    const parent = dirname(cursor);
    if (parent === cursor) return resolve(path);
    missing.unshift(basename(cursor));
    cursor = parent;
  }
  return join(canonicalExisting(cursor), ...missing);
}

function containedPath(parent: string, child: string): boolean {
  const rel = relative(parent, child);
  return rel !== "" && rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

function fileIdentity(path: string): FileIdentity {
  const info = lstatSync(path);
  return { device: info.dev, inode: info.ino };
}

function sameIdentity(left: FileIdentity, right: FileIdentity): boolean {
  return left.device === right.device && left.inode === right.inode;
}

function assertFileIdentity(path: string, expected: FileIdentity, action: string): void {
  let current: FileIdentity;
  try {
    current = fileIdentity(path);
  } catch (error) {
    throw new Error(
      `${action}: ${path}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (!sameIdentity(current, expected)) {
    throw new Error(`${action}: ${path} changed identity`);
  }
}

/**
 * Remove a validated tree without handing a recursive pathname to rm(3).
 *
 * Every descendant is identity-checked immediately before its individual
 * unlink/rmdir. In particular, the final root operation is a non-recursive
 * rmdir, so replacing the quarantine pathname with unrelated non-empty data
 * cannot make cleanup recursively delete that replacement.
 */
function removeValidatedTree(path: string, expected: FileIdentity): void {
  assertFileIdentity(path, expected, "cannot remove changed Barbarian path");
  const children = readdirSync(path, { withFileTypes: true });
  for (const child of children) {
    assertFileIdentity(path, expected, "cannot remove changed Barbarian path");
    const childPath = join(path, child.name);
    const childIdentity = fileIdentity(childPath);
    const info = lstatSync(childPath);
    if (info.isDirectory() && !info.isSymbolicLink()) {
      removeValidatedTree(childPath, childIdentity);
    } else {
      assertFileIdentity(childPath, childIdentity, "cannot remove changed Barbarian entry");
      unlinkSync(childPath);
    }
  }
  assertFileIdentity(path, expected, "cannot remove changed Barbarian path");
  rmdirSync(path);
}

function assertBaseOutsideRepo(base: string, repo: string): void {
  if (samePath(base, repo) || containedPath(repo, base)) {
    throw new Error(`barbarian directory must be outside the reviewed repository: ${base}`);
  }
}

/** Resolve the shared directory containing Barbarian run directories. */
export function barbarianDir(override?: string): string {
  const selected = override?.trim() || process.env.RATH_BARBARIAN_DIR?.trim();
  return canonicalPotential(selected || join(tmpdir(), "rath-barbarian"));
}

/** Create, identify, and exclusively lock one retained Barbarian run directory. */
export function createLockedBarbarianArtifactRoot(
  mode: BarbarianMode,
  repo: string,
  override?: string,
): { artifactRoot: string; release: BarbarianRunLockRelease } {
  const requestedBase = barbarianDir(override);
  const canonicalRepo = canonicalExisting(repo);
  assertBaseOutsideRepo(requestedBase, canonicalRepo);
  mkdirSync(requestedBase, { recursive: true });
  const base = canonicalExisting(requestedBase);
  assertBaseOutsideRepo(base, canonicalRepo);
  const baseIdentity = fileIdentity(base);
  const requestedStaging = mkdtempSync(join(base, ".rath-barbarian-staging-"));
  let staging: string;
  try {
    staging = canonicalExisting(requestedStaging);
  } catch {
    throw new Error(`barbarian directory changed during run creation: ${base}`);
  }
  const stagingIdentity = fileIdentity(staging);
  const artifactRoot = join(base, `${mode}-${randomUUID()}`);
  const run: BarbarianRun = {
    version: 1,
    mode,
    repo: canonicalRepo,
    artifactRoot,
    createdAt: new Date().toISOString(),
  };
  let published = false;
  let stagingRelease: BarbarianRunLockRelease | undefined;
  try {
    writeFileSync(join(staging, RUN_FILE), `${JSON.stringify(run, null, 2)}\n`);
    stagingRelease = acquireBarbarianRunLock(staging);
    if (!sameIdentity(baseIdentity, fileIdentity(base))) {
      throw new Error(`barbarian directory changed during run creation: ${base}`);
    }
    renameSync(staging, artifactRoot);
    published = true;
    const canonicalRoot = canonicalExisting(artifactRoot);
    if (
      !sameIdentity(baseIdentity, fileIdentity(base)) ||
      !samePath(canonicalRoot, artifactRoot) ||
      !samePath(dirname(canonicalRoot), base) ||
      samePath(canonicalRoot, canonicalRepo) ||
      containedPath(canonicalRepo, canonicalRoot)
    ) {
      throw new Error(`barbarian directory changed during run creation: ${base}`);
    }
  } catch (error) {
    stagingRelease?.(published ? artifactRoot : staging);
    if (!published) {
      try {
        if (
          sameIdentity(stagingIdentity, fileIdentity(staging)) &&
          samePath(canonicalExisting(staging), staging)
        ) {
          rmSync(staging, { recursive: true, force: true });
        }
      } catch {
        // Preserve an identity-changed staging path rather than deleting by name.
      }
    }
    throw error;
  }
  const release: BarbarianRunLockRelease = (artifactRootOverride = artifactRoot) =>
    stagingRelease?.(artifactRootOverride);
  return { artifactRoot, release };
}

/** Create an inactive run directory, primarily for programmatic staging. */
export function createBarbarianArtifactRoot(
  mode: BarbarianMode,
  repo: string,
  override?: string,
): string {
  const { artifactRoot, release } = createLockedBarbarianArtifactRoot(mode, repo, override);
  release();
  return artifactRoot;
}

function readJson(path: string): unknown {
  return JSON.parse(readFileSync(path, "utf8")) as unknown;
}

function readLockOwner(path: string): ReviewLockOwner | undefined {
  try {
    const owner = readJson(path) as Partial<ReviewLockOwner>;
    return owner.version === 1 && Number.isSafeInteger(owner.pid)
      ? (owner as ReviewLockOwner)
      : undefined;
  } catch {
    return undefined;
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return errno(error) === "EPERM";
  }
}

function activeLockError(lock: string): BarbarianRunActiveError | undefined {
  const owner = readLockOwner(join(lock, "owner.json"));
  if (owner) {
    if (owner.hostname !== hostname()) {
      return new BarbarianRunActiveError(
        `barbarian review is already active on ${owner.hostname} (pid ${owner.pid})`,
      );
    }
    if (processIsAlive(owner.pid)) {
      return new BarbarianRunActiveError(`barbarian review is already active (pid ${owner.pid})`);
    }
    return undefined;
  }
  try {
    return Date.now() - statSync(lock).mtimeMs < LOCK_INITIALIZATION_GRACE_MS
      ? new BarbarianRunActiveError("barbarian review lock is currently being initialized")
      : undefined;
  } catch {
    return undefined;
  }
}

function activeDisplacedLockError(artifactRoot: string): BarbarianRunActiveError | undefined {
  let entries: Dirent[];
  try {
    entries = readdirSync(artifactRoot, { withFileTypes: true });
  } catch {
    return undefined;
  }
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith(`${LOCK_DIR}.stale-`)) continue;
    const error = activeLockError(join(artifactRoot, entry.name));
    if (error) return error;
  }
  return undefined;
}

function restoreDisplacedLock(stale: string, lock: string): void {
  for (let attempt = 0; attempt < 200; attempt++) {
    try {
      renameSync(stale, lock);
      return;
    } catch (error) {
      if (!new Set(["EEXIST", "ENOTEMPTY"]).has(errno(error) ?? "")) throw error;
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5);
    }
  }
  throw new Error(`could not restore a concurrently replaced Barbarian lock: ${lock}`);
}

/** Whether cleanup must conservatively treat a run as active. */
export function barbarianRunIsActive(artifactRoot: string): boolean {
  const lock = join(artifactRoot, LOCK_DIR);
  return (
    activeLockError(lock) !== undefined || activeDisplacedLockError(artifactRoot) !== undefined
  );
}

/** Acquire exclusive ownership of one persisted review. */
export function acquireBarbarianRunLock(artifactRoot: string): BarbarianRunLockRelease {
  const lock = join(artifactRoot, LOCK_DIR);
  const ownerPath = join(lock, "owner.json");
  const localHostname = hostname();
  const owner: ReviewLockOwner = {
    version: 1,
    hostname: localHostname,
    pid: process.pid,
    token: randomUUID(),
    startedAt: new Date().toISOString(),
  };
  for (;;) {
    const displacedError = activeDisplacedLockError(artifactRoot);
    if (displacedError) throw displacedError;
    try {
      mkdirSync(lock);
      const createdLockIdentity = fileIdentity(lock);
      try {
        writeFileSync(ownerPath, `${JSON.stringify(owner, null, 2)}\n`, { flag: "wx" });
      } catch (error) {
        // A failed owner write leaves an initialization lock for the normal
        // grace-period recovery path. Do not delete by pathname here: another
        // contender may have replaced the directory after mkdir succeeded.
        try {
          assertFileIdentity(lock, createdLockIdentity, "Barbarian lock changed during setup");
        } catch {
          // Preserve the replacement lock; the original write error is the
          // useful failure for this acquisition attempt.
        }
        throw error;
      }
      const displacedAfterAcquisition = activeDisplacedLockError(artifactRoot);
      if (displacedAfterAcquisition) {
        if (readLockOwner(ownerPath)?.token === owner.token) {
          rmSync(lock, { recursive: true, force: true });
        }
        throw displacedAfterAcquisition;
      }
      return (artifactRootOverride = artifactRoot) => {
        let entries: Dirent[] = [];
        try {
          entries = readdirSync(artifactRootOverride, { withFileTypes: true });
        } catch {
          // The artifact root may already have been removed by successful cleanup.
        }
        const lockNames = [
          LOCK_DIR,
          ...entries
            .filter((entry) => entry.isDirectory() && entry.name.startsWith(`${LOCK_DIR}.stale-`))
            .map((entry) => entry.name),
        ];
        for (const lockName of lockNames) {
          const releaseLock = join(artifactRootOverride, lockName);
          if (readLockOwner(join(releaseLock, "owner.json"))?.token === owner.token) {
            rmSync(releaseLock, { recursive: true, force: true });
          }
        }
      };
    } catch (error) {
      if (errno(error) !== "EEXIST") throw error;
    }

    let observedIdentity: FileIdentity;
    try {
      observedIdentity = fileIdentity(lock);
    } catch (error) {
      if (errno(error) === "ENOENT") continue;
      throw error;
    }
    const current = readLockOwner(ownerPath);
    if (current) {
      if (current.hostname !== localHostname) {
        throw new BarbarianRunActiveError(
          `barbarian review is already active on ${current.hostname} (pid ${current.pid})`,
        );
      }
      if (processIsAlive(current.pid)) {
        throw new BarbarianRunActiveError(
          `barbarian review is already active (pid ${current.pid})`,
        );
      }
    } else {
      let age = 0;
      try {
        age = Date.now() - statSync(lock).mtimeMs;
      } catch (error) {
        if (errno(error) === "ENOENT") continue;
        throw error;
      }
      if (age < LOCK_INITIALIZATION_GRACE_MS) {
        throw new BarbarianRunActiveError("barbarian review lock is currently being initialized");
      }
    }

    const stale = `${lock}.stale-${randomUUID()}`;
    try {
      renameSync(lock, stale);
    } catch (error) {
      if (errno(error) === "ENOENT") continue;
      throw error;
    }
    const movedIdentity = fileIdentity(stale);
    if (!sameIdentity(observedIdentity, movedIdentity)) {
      const replacementError = activeLockError(stale);
      restoreDisplacedLock(stale, lock);
      throw replacementError ?? new BarbarianRunActiveError("barbarian review lock changed");
    }
    rmSync(stale, { recursive: true, force: true });
  }
}

function runGit(repo: string, args: string[]): string {
  try {
    return execFileSync("git", ["-C", repo, ...args], {
      encoding: "utf8",
      maxBuffer: 1024 * 1024 * 1024,
      env: gitEnvironment(),
    }).trim();
  } catch (error) {
    throw new Error(
      `git -C ${repo} ${args.join(" ")} failed: ${error instanceof Error ? error.message : error}`,
    );
  }
}

function listWorktrees(repo: string): WorktreeRecord[] {
  let output: string;
  try {
    output = execFileSync("git", ["-C", repo, "worktree", "list", "--porcelain", "-z"], {
      encoding: "utf8",
      maxBuffer: 1024 * 1024 * 1024,
      env: gitEnvironment(),
    });
  } catch (error) {
    throw new Error(
      `git -C ${repo} worktree list --porcelain -z failed: ${
        error instanceof Error ? error.message : error
      }`,
    );
  }
  if (!output) return [];
  return output.split("\0\0").flatMap((record) => {
    const fields = record.split("\0");
    const path = fields.find((field) => field.startsWith("worktree "))?.slice("worktree ".length);
    return path
      ? [{ path: resolve(path), prunable: fields.some((field) => field.startsWith("prunable ")) }]
      : [];
  });
}

function runRootForWorktree(
  path: string,
  base: string,
  includeLegacy: boolean,
): string | undefined {
  if (containedPath(base, path)) {
    const first = relative(base, path).split(sep)[0];
    if (first && NEW_RUN.test(first)) return join(base, first);
  }
  if (!includeLegacy) return undefined;
  const temporary = canonicalExisting(tmpdir());
  if (!containedPath(temporary, path)) return undefined;
  const segments = relative(temporary, path).split(sep);
  const rootName = segments[0] && LEGACY_RUN.test(segments[0]) ? segments[0] : undefined;
  return rootName ? join(temporary, rootName) : undefined;
}

function rootIsAllowed(artifactRoot: string, base: string, includeLegacy: boolean): boolean {
  if (
    samePath(dirname(artifactRoot), base) &&
    NEW_RUN.test(basename(artifactRoot)) &&
    containedPath(base, artifactRoot)
  ) {
    return true;
  }
  const temporary = canonicalExisting(tmpdir());
  return (
    includeLegacy &&
    samePath(dirname(artifactRoot), temporary) &&
    LEGACY_RUN.test(basename(artifactRoot))
  );
}

function ownerFrom(path: string): Record<string, unknown> | undefined {
  try {
    const value = readJson(path);
    return typeof value === "object" && value !== null
      ? (value as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

function expectedMode(artifactRoot: string): BarbarianMode {
  return basename(artifactRoot).includes("horde-") ? "horde" : "solo";
}

function ownerIsValid(
  owner: Record<string, unknown> | undefined,
  artifactRoot: string,
  repo: string,
  legacy: boolean,
): boolean {
  if (!owner || typeof owner.repo !== "string" || typeof owner.artifactRoot !== "string") {
    return false;
  }
  const mode = expectedMode(artifactRoot);
  if (legacy) {
    if (mode === "solo" && !legacySoloCheckpointIsComplete(owner)) return false;
    if (mode === "horde" && !legacyHordeCheckpointIsComplete(owner)) return false;
  } else if (owner.version !== 1 || owner.mode !== mode || typeof owner.createdAt !== "string") {
    return false;
  }
  let ownerRepo: string;
  let ownerRoot: string;
  try {
    ownerRepo = canonicalExisting(owner.repo);
    ownerRoot = canonicalExisting(owner.artifactRoot);
  } catch {
    return false;
  }
  return samePath(ownerRepo, repo) && samePath(ownerRoot, artifactRoot);
}

function optionalString(value: unknown): boolean {
  return value === undefined || typeof value === "string";
}

function stringArray(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function legacySoloCheckpointIsComplete(owner: Record<string, unknown>): boolean {
  return (
    owner.version === 1 &&
    typeof owner.source === "string" &&
    typeof owner.target === "string" &&
    optionalString(owner.syntheticTarget) &&
    typeof owner.modelSpec === "string" &&
    typeof owner.reasoning === "string" &&
    Array.isArray(owner.messages)
  );
}

function legacyHordeCheckpointIsComplete(owner: Record<string, unknown>): boolean {
  return (
    owner.version === 2 &&
    owner.mode === "horde" &&
    typeof owner.source === "string" &&
    typeof owner.target === "string" &&
    optionalString(owner.syntheticTarget) &&
    typeof owner.chieftainWorktree === "string" &&
    typeof owner.chieftainModelSpec === "string" &&
    typeof owner.chieftainReasoning === "string" &&
    typeof owner.hordeModelSpec === "string" &&
    typeof owner.hordeReasoning === "string" &&
    Number.isSafeInteger(owner.concurrency) &&
    (owner.concurrency as number) > 0 &&
    optionalString(owner.instructions) &&
    Array.isArray(owner.chieftainMessages) &&
    stringArray(owner.attackIds) &&
    Number.isSafeInteger(owner.nextAttackNumber) &&
    (owner.nextAttackNumber as number) > 0 &&
    Number.isSafeInteger(owner.revision) &&
    (owner.revision as number) >= 0
  );
}

function rootBelongsToRepo(artifactRoot: string, repo: string, legacy: boolean): boolean {
  let canonicalRoot: string;
  try {
    const rootInfo = lstatSync(artifactRoot);
    if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) return false;
    canonicalRoot = canonicalExisting(artifactRoot);
  } catch {
    return false;
  }
  if (
    samePath(canonicalRoot, repo) ||
    containedPath(canonicalRoot, repo) ||
    containedPath(repo, canonicalRoot)
  ) {
    return false;
  }
  const ownerFile = legacy ? CHECKPOINT_FILE : RUN_FILE;
  return ownerIsValid(ownerFrom(join(artifactRoot, ownerFile)), canonicalRoot, repo, legacy);
}

function identifyRun(
  artifactRoot: string,
  repo: string,
  base: string,
  includeLegacy: boolean,
): RunIdentity | undefined {
  if (!rootIsAllowed(artifactRoot, base, includeLegacy)) return undefined;
  const legacy = LEGACY_RUN.test(basename(artifactRoot));
  if (!rootBelongsToRepo(artifactRoot, repo, legacy)) return undefined;
  try {
    const info = lstatSync(artifactRoot);
    const canonicalRoot = canonicalExisting(artifactRoot);
    if (!samePath(canonicalRoot, artifactRoot)) return undefined;
    return {
      artifactRoot,
      canonicalRoot,
      device: info.dev,
      inode: info.ino,
    };
  } catch {
    return undefined;
  }
}

function assertRunIdentity(
  identity: RunIdentity,
  repo: string,
  base: string,
  includeLegacy: boolean,
): void {
  const current = identifyRun(identity.artifactRoot, repo, base, includeLegacy);
  if (
    !current ||
    !samePath(current.canonicalRoot, identity.canonicalRoot) ||
    current.device !== identity.device ||
    current.inode !== identity.inode
  ) {
    throw new Error(`Barbarian run changed during cleanup: ${identity.artifactRoot}`);
  }
}

function quarantineRun(identity: RunIdentity): string {
  const quarantine = join(
    dirname(identity.artifactRoot),
    `.${basename(identity.artifactRoot)}.cleaning-${randomUUID()}`,
  );
  renameSync(identity.artifactRoot, quarantine);
  try {
    const quarantinedIdentity = fileIdentity(quarantine);
    const canonicalQuarantine = canonicalExisting(quarantine);
    if (
      !sameIdentity(identity, quarantinedIdentity) ||
      !samePath(dirname(canonicalQuarantine), dirname(identity.canonicalRoot))
    ) {
      throw new Error(`Barbarian run changed while entering quarantine: ${identity.artifactRoot}`);
    }
    return canonicalQuarantine;
  } catch (error) {
    try {
      renameSync(quarantine, identity.artifactRoot);
    } catch (restoreError) {
      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; preserved at ${quarantine} because restoration failed: ${
          restoreError instanceof Error ? restoreError.message : String(restoreError)
        }`,
      );
    }
    throw error;
  }
}

function assertWorktreeContained(identity: RunIdentity, worktree: WorktreeRecord): void {
  const lexicalPath = resolve(worktree.path);
  if (!containedPath(identity.artifactRoot, lexicalPath)) {
    throw new Error(
      `Barbarian worktree is outside its run root: ${worktree.path} (root ${identity.artifactRoot})`,
    );
  }
  if (!worktree.prunable) {
    const canonicalPath = canonicalExisting(worktree.path);
    if (!containedPath(identity.canonicalRoot, canonicalPath)) {
      throw new Error(
        `Barbarian worktree resolves outside its run root: ${worktree.path} (root ${identity.artifactRoot})`,
      );
    }
  }
}

function discoverRunRoots(
  repo: string,
  base: string,
  worktrees: WorktreeRecord[],
  includeLegacy: boolean,
): string[] {
  const roots = new Set<string>();
  for (const worktree of worktrees) {
    const root = runRootForWorktree(worktree.path, base, includeLegacy);
    if (root) roots.add(root);
  }
  const scan = (parent: string, pattern: RegExp) => {
    let entries: Dirent[];
    try {
      entries = readdirSync(parent, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.isDirectory() && pattern.test(entry.name)) {
        roots.add(join(parent, entry.name));
      }
    }
  };
  scan(base, NEW_RUN);
  if (includeLegacy) scan(canonicalExisting(tmpdir()), LEGACY_RUN);
  return [...roots]
    .filter((root) => identifyRun(root, repo, base, includeLegacy) !== undefined)
    .sort((left, right) => left.localeCompare(right));
}

function nestedGitRoots(artifactRoot: string): string[] {
  const roots: string[] = [];
  const pending = [artifactRoot];
  while (pending.length > 0) {
    const current = pending.pop()!;
    let entries: Dirent[];
    try {
      entries = readdirSync(current, { withFileTypes: true });
    } catch (error) {
      throw new Error(
        `cannot inspect Barbarian run before cleanup: ${current}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
    const byName = new Map(entries.map((entry) => [entry.name, entry]));
    if (
      byName.get("HEAD")?.isFile() &&
      byName.get("objects")?.isDirectory() &&
      byName.get("refs")?.isDirectory()
    ) {
      roots.push(current);
      continue;
    }
    for (const entry of entries) {
      if (entry.name === ".git") {
        roots.push(current);
      } else if (entry.isDirectory()) {
        pending.push(join(current, entry.name));
      }
    }
  }
  return roots;
}

function unownedNestedGitRoots(identity: RunIdentity, ownedWorktrees: WorktreeRecord[]): string[] {
  const ownedPaths = ownedWorktrees.map((worktree) => resolve(worktree.path));
  return nestedGitRoots(identity.artifactRoot).filter(
    (gitRoot) => !ownedPaths.some((owned) => samePath(resolve(gitRoot), owned)),
  );
}

function snapshotRunTree(artifactRoot: string, excludedWorktrees: WorktreeRecord[] = []): string {
  const excluded = excludedWorktrees.map((worktree) => resolve(worktree.path));
  const entries: string[] = [];
  const pending = [artifactRoot];
  while (pending.length > 0) {
    const current = pending.pop()!;
    let children: Dirent[];
    try {
      children = readdirSync(current, { withFileTypes: true });
    } catch (error) {
      throw new Error(
        `cannot snapshot Barbarian run before cleanup: ${current}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
    for (const child of children) {
      if (
        samePath(current, artifactRoot) &&
        (child.name === LOCK_DIR || child.name.startsWith(`${LOCK_DIR}.stale-`))
      ) {
        continue;
      }
      const path = join(current, child.name);
      const resolvedPath = resolve(path);
      if (
        excluded.some(
          (worktree) => samePath(resolvedPath, worktree) || containedPath(worktree, resolvedPath),
        )
      ) {
        continue;
      }
      let info: ReturnType<typeof lstatSync>;
      try {
        info = lstatSync(path);
      } catch (error) {
        throw new Error(
          `Barbarian run changed while being snapshotted: ${path}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }
      const relativePath = relative(artifactRoot, path);
      if (info.isDirectory() && !info.isSymbolicLink()) {
        entries.push(`${relativePath}\0directory\0${info.dev}\0${info.ino}`);
        pending.push(path);
      } else {
        entries.push(
          `${relativePath}\0entry\0${info.dev}\0${info.ino}\0${info.size}\0${info.mtimeMs}`,
        );
      }
    }
  }
  return entries.sort().join("\n");
}

/** Remove retained, inactive Barbarian runs for one repository. */
export function cleanBarbarianRuns(options: BarbarianCleanOptions): BarbarianCleanResult {
  const repo = canonicalExisting(options.repo);
  const base = barbarianDir(options.barbarianDir);
  assertBaseOutsideRepo(base, repo);
  const includeLegacy = !options.barbarianDir?.trim() && !process.env.RATH_BARBARIAN_DIR?.trim();
  const discoveredWorktrees = listWorktrees(repo);
  const roots = discoverRunRoots(repo, base, discoveredWorktrees, includeLegacy);
  const result: BarbarianCleanResult = {
    removed: [],
    wouldRemove: [],
    skippedActive: [],
    errors: [],
  };
  const discoveredRuns = roots.flatMap((artifactRoot) => {
    const identity = identifyRun(artifactRoot, repo, base, includeLegacy);
    if (!identity) return [];
    const ownedWorktrees = discoveredWorktrees
      .filter((worktree) =>
        samePath(runRootForWorktree(worktree.path, base, includeLegacy) ?? "", artifactRoot),
      )
      .sort((left, right) => right.path.length - left.path.length);
    return [{ identity, ownedWorktrees }];
  });

  if (options.dryRun) {
    for (const { identity, ownedWorktrees } of discoveredRuns) {
      if (barbarianRunIsActive(identity.artifactRoot)) {
        result.skippedActive.push(identity.artifactRoot);
      } else {
        try {
          const unownedGitRoots = unownedNestedGitRoots(identity, ownedWorktrees);
          if (unownedGitRoots.length > 0) {
            throw new Error(
              `refusing to remove ${identity.artifactRoot}; unowned Git worktrees or repositories exist beneath it: ${unownedGitRoots.join(", ")}`,
            );
          }
          for (const worktree of ownedWorktrees) {
            assertWorktreeContained(identity, worktree);
          }
          result.wouldRemove.push({
            artifactRoot: identity.artifactRoot,
            worktrees: ownedWorktrees.map((worktree) => worktree.path),
          });
        } catch (error) {
          result.errors.push(error instanceof Error ? error.message : String(error));
        }
      }
    }
    return result;
  }

  const lockedRuns: Array<{
    identity: RunIdentity;
    release: BarbarianRunLockRelease;
    releaseRoot: string;
    ownedWorktrees: WorktreeRecord[];
    treeSnapshot?: string;
    failed: boolean;
  }> = [];
  try {
    for (const { identity } of discoveredRuns) {
      let release: BarbarianRunLockRelease;
      try {
        release = acquireBarbarianRunLock(identity.artifactRoot);
      } catch (error) {
        if (error instanceof BarbarianRunActiveError) {
          result.skippedActive.push(identity.artifactRoot);
        } else {
          result.errors.push(error instanceof Error ? error.message : String(error));
        }
        continue;
      }
      try {
        assertRunIdentity(identity, repo, base, includeLegacy);
        lockedRuns.push({
          identity,
          release,
          releaseRoot: identity.artifactRoot,
          ownedWorktrees: [],
          treeSnapshot: undefined,
          failed: false,
        });
      } catch (error) {
        release();
        result.errors.push(error instanceof Error ? error.message : String(error));
      }
    }

    const currentWorktrees = listWorktrees(repo);
    for (const locked of lockedRuns) {
      locked.ownedWorktrees = currentWorktrees
        .filter((worktree) =>
          samePath(
            runRootForWorktree(worktree.path, base, includeLegacy) ?? "",
            locked.identity.artifactRoot,
          ),
        )
        .sort((left, right) => right.path.length - left.path.length);
      try {
        assertRunIdentity(locked.identity, repo, base, includeLegacy);
        const unownedGitRoots = unownedNestedGitRoots(locked.identity, locked.ownedWorktrees);
        if (unownedGitRoots.length > 0) {
          throw new Error(
            `refusing to remove ${locked.identity.artifactRoot}; unowned Git worktrees or repositories exist beneath it: ${unownedGitRoots.join(", ")}`,
          );
        }
        locked.treeSnapshot = snapshotRunTree(locked.identity.artifactRoot, locked.ownedWorktrees);
      } catch (error) {
        locked.failed = true;
        result.errors.push(error instanceof Error ? error.message : String(error));
      }
    }

    for (const locked of lockedRuns) {
      if (locked.failed) continue;
      for (const worktree of locked.ownedWorktrees) {
        try {
          assertRunIdentity(locked.identity, repo, base, includeLegacy);
          assertWorktreeContained(locked.identity, worktree);
          runGit(repo, ["worktree", "remove", "--force", worktree.path]);
        } catch (error) {
          locked.failed = true;
          result.errors.push(error instanceof Error ? error.message : String(error));
          break;
        }
      }
    }

    const remaining = listWorktrees(repo);
    for (const locked of lockedRuns) {
      const run = {
        artifactRoot: locked.identity.artifactRoot,
        worktrees: locked.ownedWorktrees.map((worktree) => worktree.path),
      };
      if (locked.failed) continue;
      const stillRegistered = remaining.some((worktree) =>
        samePath(
          runRootForWorktree(worktree.path, base, includeLegacy) ?? "",
          locked.identity.artifactRoot,
        ),
      );
      if (stillRegistered) {
        result.errors.push(`worktrees remain registered beneath ${locked.identity.artifactRoot}`);
        continue;
      }
      try {
        assertRunIdentity(locked.identity, repo, base, includeLegacy);
        if (
          locked.treeSnapshot === undefined ||
          snapshotRunTree(locked.identity.artifactRoot, locked.ownedWorktrees) !==
            locked.treeSnapshot
        ) {
          throw new Error(
            `refusing to remove ${locked.identity.artifactRoot}; run content changed during cleanup`,
          );
        }
        const remainingGitRoots = nestedGitRoots(locked.identity.artifactRoot);
        if (remainingGitRoots.length > 0) {
          throw new Error(
            `refusing to remove ${locked.identity.artifactRoot}; Git worktrees or repositories remain beneath it: ${remainingGitRoots.join(", ")}`,
          );
        }
        const quarantine = quarantineRun(locked.identity);
        locked.releaseRoot = quarantine;
        if (snapshotRunTree(quarantine) !== locked.treeSnapshot) {
          try {
            renameSync(quarantine, locked.identity.artifactRoot);
            locked.releaseRoot = locked.identity.artifactRoot;
          } catch (restoreError) {
            throw new Error(
              `refusing to remove ${locked.identity.artifactRoot}; run content changed during quarantine and was preserved at ${quarantine}: ${
                restoreError instanceof Error ? restoreError.message : String(restoreError)
              }`,
            );
          }
          throw new Error(
            `refusing to remove ${locked.identity.artifactRoot}; run content changed during quarantine`,
          );
        }
        const quarantinedGitRoots = nestedGitRoots(quarantine);
        if (quarantinedGitRoots.length > 0) {
          try {
            renameSync(quarantine, locked.identity.artifactRoot);
            locked.releaseRoot = locked.identity.artifactRoot;
          } catch (restoreError) {
            throw new Error(
              `refusing to remove ${locked.identity.artifactRoot}; Git worktrees or repositories appeared during cleanup and the run was preserved at ${quarantine}: ${
                restoreError instanceof Error ? restoreError.message : String(restoreError)
              }`,
            );
          }
          throw new Error(
            `refusing to remove ${locked.identity.artifactRoot}; Git worktrees or repositories appeared during cleanup: ${quarantinedGitRoots.join(", ")}`,
          );
        }
        if (existsSync(locked.identity.artifactRoot)) {
          throw new Error(
            `refusing to report ${locked.identity.artifactRoot} removed because the public path reappeared during cleanup; the original run was preserved at ${quarantine}`,
          );
        }
        removeValidatedTree(quarantine, locked.identity);
        result.removed.push(run);
      } catch (error) {
        result.errors.push(error instanceof Error ? error.message : String(error));
      }
    }
  } finally {
    for (const locked of lockedRuns) locked.release(locked.releaseRoot);
  }

  return result;
}
