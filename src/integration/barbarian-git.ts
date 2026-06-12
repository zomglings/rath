/**
 * Integration test: the Barbarian Reviewer's git plumbing. No API, no key.
 *
 * Covers the pieces that prepare a review range without running the agent:
 * repo-root resolution, the main->master source fallback, change detection,
 * and the synthetic target commit (staged + unstaged + untracked captured in
 * a disposable worktree, with the user's tree untouched).
 *
 * Exits 0 on success, 1 on failure.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createSyntheticTarget, hasChanges, repoRoot, resolveSource } from "../agents/barbarian.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" }).trim();
}

function initRepo(dir: string, branch: string): void {
  execFileSync("git", ["init", "-b", branch, dir], { stdio: "ignore" });
  git(dir, "config", "user.email", "test@example.com");
  git(dir, "config", "user.name", "Test");
}

async function main(): Promise<void> {
  // realpath: git reports physical paths (macOS /var -> /private/var).
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "rath-barbarian-git-")));
  try {
    // Case 1: repoRoot resolves from a subdirectory; rejects a non-repo.
    const repo = join(dir, "repo");
    initRepo(repo, "main");
    writeFileSync(join(repo, "a.txt"), "one\n");
    git(repo, "add", "a.txt");
    git(repo, "commit", "-m", "initial");
    const sub = join(repo, "sub");
    mkdirSync(sub);
    assert.equal(repoRoot(sub), repo, "repoRoot resolves from a subdirectory");
    assert.throws(() => repoRoot(dir), /not in a git work tree/, "non-repo rejected");
    log("Case 1 OK: repoRoot (subdirectory, non-repo)");

    // Case 2: resolveSource — explicit wins; main preferred; master fallback;
    // neither is an error.
    assert.equal(resolveSource(repo, "HEAD~0"), "HEAD~0", "explicit source wins");
    assert.equal(resolveSource(repo), "main", "main preferred");
    const masterRepo = join(dir, "master-repo");
    initRepo(masterRepo, "master");
    writeFileSync(join(masterRepo, "a.txt"), "one\n");
    git(masterRepo, "add", "a.txt");
    git(masterRepo, "commit", "-m", "initial");
    assert.equal(resolveSource(masterRepo), "master", "master fallback");
    const branchRepo = join(dir, "branch-repo");
    initRepo(branchRepo, "trunk");
    writeFileSync(join(branchRepo, "a.txt"), "one\n");
    git(branchRepo, "add", "a.txt");
    git(branchRepo, "commit", "-m", "initial");
    assert.throws(() => resolveSource(branchRepo), /neither main nor master/, "no fallback found");
    log("Case 2 OK: resolveSource (explicit, main, master, neither)");

    // Case 3: hasChanges — clean, then each kind of change.
    assert.equal(hasChanges(repo), false, "clean tree has no changes");
    writeFileSync(join(repo, "untracked.txt"), "new\n");
    assert.equal(hasChanges(repo), true, "untracked file counts");
    rmSync(join(repo, "untracked.txt"));
    writeFileSync(join(repo, "a.txt"), "two\n");
    assert.equal(hasChanges(repo), true, "unstaged edit counts");
    git(repo, "checkout", "--", "a.txt");
    log("Case 3 OK: hasChanges (clean, untracked, unstaged)");

    // Case 4: createSyntheticTarget captures staged + unstaged + untracked,
    // and leaves the user's tree exactly as it was.
    writeFileSync(join(repo, "staged.txt"), "staged\n");
    git(repo, "add", "staged.txt");
    writeFileSync(join(repo, "a.txt"), "edited\n"); // unstaged
    mkdirSync(join(repo, "deep"), { recursive: true });
    writeFileSync(join(repo, "deep", "untracked.txt"), "untracked\n");

    const before = git(repo, "status", "--porcelain=v1", "-uall");
    const artifactRoot = mkdtempSync(join(tmpdir(), "rath-barbarian-artifacts-"));
    const sha = createSyntheticTarget(repo, artifactRoot);
    assert.match(sha, /^[0-9a-f]{40}$/, "synthetic target is a full SHA");

    // The commit is reachable from the main repo (shared object store).
    const show = (path: string) => git(repo, "show", `${sha}:${path}`);
    assert.equal(show("staged.txt"), "staged", "staged change captured");
    assert.equal(show("a.txt"), "edited", "unstaged change captured");
    assert.equal(show("deep/untracked.txt"), "untracked", "untracked file captured");

    // The user's tree is untouched: same status, same file contents.
    assert.equal(
      git(repo, "status", "--porcelain=v1", "-uall"),
      before,
      "user's working tree status unchanged",
    );
    assert.equal(readFileSync(join(repo, "a.txt"), "utf8"), "edited\n", "tree contents intact");

    // The diff source..synthetic contains exactly the captured changes.
    const diffStat = git(repo, "diff", "--name-only", `main..${sha}`).split("\n").sort();
    assert.deepEqual(
      diffStat,
      ["a.txt", "deep/untracked.txt", "staged.txt"],
      "diff covers exactly the working-tree changes",
    );
    log("Case 4 OK: createSyntheticTarget (staged+unstaged+untracked, tree untouched)");

    // Cleanup: prune the worktree so the temp dirs can be removed.
    git(repo, "worktree", "remove", "--force", join(artifactRoot, "current-state"));
    rmSync(artifactRoot, { recursive: true, force: true });

    // Case 5: raw-bytes regression (the barbarian's own finding). Patch data
    // must not be trimmed — an unstaged edit whose only change is trailing
    // whitespace on the last line dies in `git apply` if the diff was trimmed
    // — and an untracked filename with a leading space must survive the
    // ls-files -z parse.
    const rawRepo = join(dir, "raw-repo");
    initRepo(rawRepo, "main");
    writeFileSync(join(rawRepo, "f.txt"), "base\n");
    git(rawRepo, "add", "f.txt");
    git(rawRepo, "commit", "-m", "initial");
    writeFileSync(join(rawRepo, "f.txt"), "base   \n"); // trailing spaces only
    writeFileSync(join(rawRepo, " leading.txt"), "space\n"); // leading-space name
    const rawArtifacts = mkdtempSync(join(tmpdir(), "rath-barbarian-raw-"));
    const rawSha = createSyntheticTarget(rawRepo, rawArtifacts);
    assert.equal(
      git(rawRepo, "show", `${rawSha}:f.txt`),
      "base", // git() trims display output; the commit itself is exact —
      "trailing-whitespace-only edit captured",
    );
    assert.equal(
      execFileSync("git", ["-C", rawRepo, "show", `${rawSha}:f.txt`], { encoding: "utf8" }),
      "base   \n",
      "trailing whitespace preserved byte-for-byte",
    );
    assert.equal(
      execFileSync("git", ["-C", rawRepo, "show", `${rawSha}: leading.txt`], {
        encoding: "utf8",
      }),
      "space\n",
      "leading-space filename captured",
    );
    git(rawRepo, "worktree", "remove", "--force", join(rawArtifacts, "current-state"));
    rmSync(rawArtifacts, { recursive: true, force: true });
    log("Case 5 OK: raw patch bytes and -z lists are not trimmed");
    log("barbarian-git: all cases passed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error);
    process.exit(1);
  },
);
