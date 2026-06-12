/**
 * The Barbarian Reviewer: a non-interactive subagent that reviews the changes
 * from a source commit-ish to a target commit-ish in a git repository and
 * returns a findings report (its final message; the caller prints it).
 *
 * This module is the whole agent: the system prompt, the git plumbing that
 * prepares the review range (including a synthetic commit capturing the
 * working tree when no target is given), and the agent loop that runs the
 * review in-process. It is deliberately frontend-free — the `rath barbarian
 * run` command (src/commands/barbarian.ts) is a thin wrapper over
 * runBarbarianReview. The barbarian is its own program: anything that wants a
 * review (a human, or an agent in `rath run` via bash) invokes that command.
 *
 * The caller chooses the barbarian's model and reasoning level; they default
 * to the pinned default model and "high" (reviews want depth, and the
 * barbarian runs unattended where latency is cheap).
 */
import { spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import {
  type AgentContext,
  type AgentEvent,
  type AgentMessage,
  agentLoop,
  type StreamFn,
} from "@earendil-works/pi-agent-core";
import {
  type AssistantMessage,
  type Message,
  type SimpleStreamOptions,
  streamSimple,
} from "@earendil-works/pi-ai";
import { loadPreferences } from "../config.js";
import { DEFAULT_DEFAULT_MODEL, type ReasoningLevel, resolveModel } from "../models.js";

export const BARBARIAN_SYSTEM_PROMPT = `You are the Barbarian Reviewer. You run non-interactively inside a git repository. Your task is to review the changes from a source commit-ish to a target commit-ish. Never ask questions. Never request approval. Finish with a complete findings report.

You are not a linter. Ignore formatting, style-only, naming-only, and syntactic nits unless they hide a real defect. Attack defects: correctness, regressions, broken contracts, data loss, races, security exposure, performance collapse, bad migrations, bad tests, bad error handling, bad UX, operational hazards, maintainability traps, dependency/API misuse, and incomplete changes.

Be relentless. Be skeptical. Read outside the diff when necessary. The diff is the attack surface, not the whole battlefield.

Inputs are supplied in the initial user message:

- \`SOURCE\`: source commit-ish
- \`TARGET\`: target commit-ish
- \`ARTIFACT_ROOT\`: directory for disposable worktrees, repro scripts, logs, and other review artifacts

Operating procedure:

1. Verify you are in a git work tree.
2. Resolve both inputs with \`git rev-parse --verify\` and state the resolved SHAs.
3. Inspect the patch with \`git diff --stat SOURCE..TARGET\` and \`git diff --find-renames SOURCE..TARGET\`.
4. Identify changed files, then read enough unchanged surrounding code to understand contracts and call paths. Use grep/find/read/bash as needed.
5. For each suspected issue, try to prove it. Prefer concrete reproduction over speculation.
6. When a reproduction is useful, create a disposable worktree for \`TARGET\` under \`ARTIFACT_ROOT\`, for example:

   \`\`\`sh
   mkdir -p "$ARTIFACT_ROOT"
   git worktree add --detach "$ARTIFACT_ROOT/repro-name" TARGET
   \`\`\`

   Put repro scripts, fixtures, or notes in that worktree. Run the minimal command that demonstrates the defect. Capture exact commands and relevant output. You may create multiple worktrees.
7. Clean up worktrees only if cleanup will not destroy useful artifacts the user should inspect. If leaving artifacts, state exact paths and why.
8. Do not modify the user's working tree. Confine all writes to ARTIFACT_ROOT.

Finding standard:

- Report only issues with a plausible path to user-visible failure, production risk, broken tests, or significant future maintenance cost.
- Do not report lint/format/test-coverage preferences as findings.
- If a finding depends on an assumption, name the assumption and say how to verify it.
- Prefer fewer, sharper findings over a bag of weak complaints.

Final response contract:

Your final assistant message IS the findings report — it is printed as the result of the review. It must contain the report only. Do not wrap it in Markdown fences. Do not include chatter.

Output format:

\`\`\`text
Reviewed: <source-sha>..<target-sha>

Findings:

1. <severity>: <plain defect statement>
   Evidence: <file:line/function, command output, or reasoning chain>
   Reproduction: <exact command(s), worktree path if any, or "not staged">
   Fix: <specific required change>

2. ...

No finding: <only if no findings survived verification>

Artifacts:
- <paths to worktrees/repro scripts/logs, or "none">
\`\`\`

Severity vocabulary: \`blocker\`, \`major\`, \`minor\`. Do not invent softer labels.

Tone:

- Terse.
- Plain.
- Unforgiving.
- No praise.
- No hedging unless the evidence is actually uncertain.`;

/**
 * Injected to continue a review whose previous turn ended in a provider error
 * or content filter, instead of discarding the whole run. Also nudges toward
 * neutral phrasing, since security findings phrased as exploit payloads are a
 * common (and transient) content-filter trigger.
 */
const BARBARIAN_CONTINUE_NUDGE =
  "Your previous turn was cut off by a provider error or content filter. " +
  "Continue and complete the review. State each finding neutrally — name the " +
  "defect and its impact, but do not reproduce exploit payloads or step-by-step " +
  "attack instructions. End with your complete findings report.";

/** Injected when resuming a checkpointed review so the agent picks up cleanly. */
const BARBARIAN_RESUME_NUDGE =
  "This review was interrupted and is being resumed from a checkpoint. Pick up " +
  "where you left off: finish any open investigation, then end with your " +
  "complete findings report.";

const CHECKPOINT_FILE = "checkpoint.json";

/**
 * On-disk checkpoint of an in-progress review, written to the artifact root
 * after each turn so a review that dies (rate limit, crash, interrupt) can be
 * resumed instead of restarted.
 */
interface BarbarianCheckpoint {
  version: 1;
  repo: string;
  source: string;
  target: string;
  syntheticTarget?: string;
  modelSpec: string;
  reasoning: ReasoningLevel;
  artifactRoot: string;
  messages: AgentMessage[];
}

/**
 * True when a resumable checkpoint exists at the artifact root. The CLI uses
 * this to decide whether to suggest `--resume` on failure: a pre-flight failure
 * (bad model spec, synthetic-target error) leaves an artifact root with no
 * checkpoint, and resuming it would only fail with "no barbarian checkpoint".
 */
export function hasCheckpoint(artifactRoot: string): boolean {
  return existsSync(join(artifactRoot, CHECKPOINT_FILE));
}

function loadCheckpoint(artifactRoot: string): BarbarianCheckpoint {
  const path = join(artifactRoot, CHECKPOINT_FILE);
  if (!existsSync(path)) {
    throw new Error(`no barbarian checkpoint at ${path}`);
  }
  let parsed: BarbarianCheckpoint;
  try {
    parsed = JSON.parse(readFileSync(path, "utf8")) as BarbarianCheckpoint;
  } catch (error) {
    throw new Error(
      `${path} is not a valid barbarian checkpoint (unreadable JSON): ${
        error instanceof Error ? error.message : error
      }`,
    );
  }
  if (parsed.version !== 1 || !Array.isArray(parsed.messages)) {
    throw new Error(`${path} is not a valid barbarian checkpoint`);
  }
  return parsed;
}

export interface BarbarianOptions {
  /** Git repository (any path inside it). Defaults to the process cwd. */
  repo?: string;
  /** Source commit-ish. Defaults to main, falling back to master. */
  source?: string;
  /**
   * Target commit-ish. Defaults to the current repository state — including
   * staged, unstaged, and untracked changes — captured as a synthetic commit
   * in a disposable worktree (the user's tree is never touched).
   */
  target?: string;
  /** Extra reviewer instructions appended to the prompt. */
  instructions?: string;
  /** Model spec <provider>/<model-id>. Defaults to the pinned default model. */
  model?: string;
  /** Reasoning effort. Defaults to "high". */
  reasoning?: ReasoningLevel;
  /** Observer for agent events (progress reporting). */
  onEvent?: (event: AgentEvent) => void;
  /**
   * Called once with the artifact root as soon as it is known — BEFORE the
   * first turn — so the caller can report the --resume path even if the review
   * later fails (the whole point of the checkpoint is to resume a failed run).
   */
  onArtifactRoot?: (artifactRoot: string) => void;
  /** Abort signal. When it fires, the agent loop stops. */
  signal?: AbortSignal;
  /**
   * Resume a prior, interrupted review from its artifact root (the directory
   * printed as "artifacts:" on the original run). The checkpointed transcript
   * and range are reloaded and the review continues. Other options that the
   * checkpoint already supplies (repo/source/target) are ignored; model and
   * reasoning may be overridden.
   */
  resume?: string;
}

export interface BarbarianResult {
  repo: string;
  source: string;
  target: string;
  artifactRoot: string;
  /** The findings report (the agent's final message). */
  findings: string;
  /** Set when the target was synthesized from the working tree. */
  syntheticTarget?: string;
  modelSpec: string;
  reasoning: ReasoningLevel;
}

/**
 * Run git and return RAW stdout. Do not trim here: patch bytes (`git diff
 * --binary`) and NUL-delimited path lists (`ls-files -z`) are data, where
 * trimming corrupts trailing whitespace in the last patch line or a leading
 * space in the first filename. Scalar outputs (SHAs, status checks) trim at
 * the call site via runGitScalar.
 */
function runGit(repo: string, args: string[], input?: string): string {
  const result = spawnSync("git", ["-C", repo, ...args], {
    encoding: "utf8",
    input,
    env: {
      ...process.env,
      // The synthetic commit must succeed on machines with no git identity.
      GIT_AUTHOR_NAME: process.env.GIT_AUTHOR_NAME ?? "Barbarian Reviewer",
      GIT_AUTHOR_EMAIL: process.env.GIT_AUTHOR_EMAIL ?? "barbarian@rath.invalid",
      GIT_COMMITTER_NAME: process.env.GIT_COMMITTER_NAME ?? "Barbarian Reviewer",
      GIT_COMMITTER_EMAIL: process.env.GIT_COMMITTER_EMAIL ?? "barbarian@rath.invalid",
    },
  });
  if (result.status !== 0) {
    throw new Error(`git -C ${repo} ${args.join(" ")} failed:\n${result.stderr || result.stdout}`);
  }
  return result.stdout;
}

/** runGit for single-value outputs (SHAs, refs): trailing newline trimmed. */
function runGitScalar(repo: string, args: string[]): string {
  return runGit(repo, args).trim();
}

/** Repo root for any path inside a work tree; throws when outside one. */
export function repoRoot(path: string): string {
  const result = spawnSync("git", ["-C", path, "rev-parse", "--show-toplevel"], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`${path} is not in a git work tree`);
  }
  return result.stdout.trim();
}

/** Default source: main if it exists, else master. */
export function resolveSource(repo: string, source?: string): string {
  if (source) {
    return source;
  }
  for (const candidate of ["main", "master"]) {
    const probe = spawnSync(
      "git",
      ["-C", repo, "rev-parse", "--verify", "--quiet", `${candidate}^{commit}`],
      { encoding: "utf8" },
    );
    if (probe.status === 0) {
      return candidate;
    }
  }
  throw new Error("no source given, and neither main nor master exists");
}

/** True when the repo has staged, unstaged, or untracked changes. */
export function hasChanges(repo: string): boolean {
  return runGitScalar(repo, ["status", "--porcelain=v1", "-uall"]).length > 0;
}

/**
 * Capture the current state of the repository — staged, unstaged, and
 * untracked changes — as a commit the reviewer can diff against, without
 * touching the user's tree. A detached worktree is created under
 * `artifactRoot`, the staged and unstaged patches are applied, untracked
 * files are copied in, and the lot is committed. Returns the commit SHA
 * (worktrees share the object store, so it resolves from the main repo).
 */
export function createSyntheticTarget(repo: string, artifactRoot: string): string {
  const worktree = join(artifactRoot, "current-state");
  runGit(repo, ["worktree", "add", "--detach", worktree, "HEAD"]);
  // Patch bytes pass through untouched (see runGit); emptiness is tested on
  // a trimmed copy only.
  const staged = runGit(repo, ["diff", "--binary", "--cached"]);
  if (staged.trim()) {
    runGit(worktree, ["apply", "--index", "--binary", "-"], staged);
  }
  const unstaged = runGit(repo, ["diff", "--binary"]);
  if (unstaged.trim()) {
    runGit(worktree, ["apply", "--binary", "-"], unstaged);
  }
  const untracked = runGit(repo, ["ls-files", "--others", "--exclude-standard", "-z"]);
  for (const rel of untracked.split("\0").filter(Boolean)) {
    const to = join(worktree, rel);
    mkdirSync(dirname(to), { recursive: true });
    cpSync(join(repo, rel), to, { dereference: false, preserveTimestamps: true });
  }
  runGit(worktree, ["add", "-A"]);
  runGit(worktree, ["commit", "--no-gpg-sign", "-m", "Synthetic barbarian target"]);
  return runGitScalar(worktree, ["rev-parse", "HEAD"]);
}

function finalText(message: AssistantMessage): string {
  return message.content
    .filter((block): block is { type: "text"; text: string } => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}

/**
 * Run a barbarian review. Resolves the repo, source, and target (synthesizing
 * a target commit from the working tree when none is given), runs the agent
 * loop to completion, and returns the findings report (the agent's final
 * message). The report is not written to a file — the caller prints it.
 *
 * The caller must have registered the providers (registerOpenAINative /
 * registerOpenRouterNative) and primed the catalogue if openrouter-native
 * model specs should resolve against the live list.
 */
export async function runBarbarianReview(options: BarbarianOptions): Promise<BarbarianResult> {
  let repo: string;
  let source: string;
  let target: string;
  let syntheticTarget: string | undefined;
  let modelSpec: string;
  let reasoning: ReasoningLevel;
  let artifactRoot: string;
  // The accumulated transcript and the prompt for the first loop iteration.
  // A resume seeds the transcript from the checkpoint and prompts a resume
  // nudge; a fresh review starts empty with the review prompt.
  let transcript: AgentMessage[];
  let promptMessages: AgentMessage[];

  if (options.resume) {
    const checkpoint = loadCheckpoint(options.resume);
    repo = checkpoint.repo;
    source = checkpoint.source;
    target = checkpoint.target;
    syntheticTarget = checkpoint.syntheticTarget;
    modelSpec = options.model?.trim() || checkpoint.modelSpec;
    reasoning = options.reasoning ?? checkpoint.reasoning;
    artifactRoot = checkpoint.artifactRoot;
    transcript = checkpoint.messages;
    promptMessages = [{ role: "user", content: BARBARIAN_RESUME_NUDGE, timestamp: Date.now() }];
  } else {
    const cwd = process.cwd();
    const repoArg = options.repo?.trim() || cwd;
    repo = repoRoot(isAbsolute(repoArg) ? repoArg : resolve(cwd, repoArg));
    modelSpec = options.model?.trim() || loadPreferences().defaultModel || DEFAULT_DEFAULT_MODEL;
    reasoning = options.reasoning ?? "high";
    artifactRoot = mkdtempSync(join(tmpdir(), "rath-barbarian-"));
    source = resolveSource(repo, options.source?.trim() || undefined);
    const requestedTarget = options.target?.trim();
    if (!requestedTarget) {
      syntheticTarget = hasChanges(repo) ? createSyntheticTarget(repo, artifactRoot) : undefined;
      target = syntheticTarget ?? "HEAD";
    } else {
      target = requestedTarget;
    }
    const instructions = options.instructions?.trim();
    const prompt = `SOURCE: ${source}
TARGET: ${target}
ARTIFACT_ROOT: ${artifactRoot}

Review SOURCE..TARGET. Read outside the diff when needed. Stage reproductions in disposable worktrees under ARTIFACT_ROOT when possible. Your final assistant message is the complete findings report. Do not ask questions. Do not wait for approval.${
      instructions ? `\n\nExtra instructions:\n${instructions}` : ""
    }`;
    transcript = [];
    promptMessages = [{ role: "user", content: prompt, timestamp: Date.now() }];
  }
  // Surface the artifact root before the first turn so the caller can print the
  // --resume path even if the review later fails — resuming a failed run is the
  // whole point of the checkpoint.
  options.onArtifactRoot?.(artifactRoot);
  const model = resolveModel(modelSpec);

  // After every turn, persist the transcript and range so an interrupted review
  // (rate limit, crash, abort) can be resumed from this artifact root rather
  // than restarted. Best-effort: a write failure must not break the review.
  const writeCheckpoint = (messages: AgentMessage[]): void => {
    try {
      const checkpoint: BarbarianCheckpoint = {
        version: 1,
        repo,
        source,
        target,
        ...(syntheticTarget !== undefined && { syntheticTarget }),
        modelSpec,
        reasoning,
        artifactRoot,
        messages,
      };
      writeFileSync(join(artifactRoot, CHECKPOINT_FILE), JSON.stringify(checkpoint, null, 2));
    } catch {
      // best-effort
    }
  };

  // The barbarian's tools run with the repo as cwd. It gets the read/search
  // set plus bash (reproductions) and write (repro scripts under the
  // artifact root; the prompt forbids writes elsewhere).
  const pi = await import("@earendil-works/pi-coding-agent");
  const context: AgentContext = {
    systemPrompt: BARBARIAN_SYSTEM_PROMPT,
    messages: [],
    tools: [
      pi.createReadTool(repo),
      pi.createBashTool(repo),
      pi.createGrepTool(repo),
      pi.createFindTool(repo),
      pi.createLsTool(repo),
      pi.createWriteTool(repo),
    ],
  };

  const loopConfig = {
    model,
    // pi-ai's stream-level reasoning has no "off" member; omitting the option
    // is how reasoning is disabled.
    ...(reasoning !== "off" && { reasoning }),
    // Drop failed turns and their orphaned tool results so a transient provider
    // error mid-review does not poison every later request. Contract: no throw.
    convertToLlm: (messages: AgentMessage[]) => {
      const droppedToolCallIds = new Set<string>();
      for (const m of messages) {
        if (m.role === "assistant" && (m.stopReason === "error" || m.stopReason === "aborted")) {
          for (const block of m.content) {
            if (block.type === "toolCall") {
              droppedToolCallIds.add(block.id);
            }
          }
        }
      }
      return messages.filter((m): m is Message => {
        if (m.role === "user") {
          return true;
        }
        if (m.role === "toolResult") {
          return !droppedToolCallIds.has(m.toolCallId);
        }
        return m.role === "assistant" && m.stopReason !== "error" && m.stopReason !== "aborted";
      });
    },
  };
  // Hosted web search stays off: the barbarian reviews code, and a
  // prompt-injectable hosted tool has no place in an unattended run.
  const streamFn: StreamFn = (m, ctx, opts) =>
    streamSimple(m, ctx, { ...opts, webSearch: false } as SimpleStreamOptions);

  // Run the review, surviving a single errored/content-filtered turn rather
  // than discarding the whole review: drop that turn (convertToLlm already
  // strips it from the request) and continue with a neutralizing nudge, up to
  // MAX_CONTINUES times. Abort is honored immediately (it is a real cancel).
  const MAX_CONTINUES = 2;
  let final: AssistantMessage | undefined;
  for (let attempt = 0; ; attempt++) {
    const events = agentLoop(
      promptMessages,
      { ...context, messages: transcript },
      loopConfig,
      options.signal,
      streamFn,
    );
    // Mirror the transcript as turns complete so the checkpoint is written
    // after EACH turn (agentLoop.result() only resolves at the end of the
    // whole invocation). live === transcript once produced is folded in below.
    const live: AgentMessage[] = [...transcript, ...promptMessages];
    writeCheckpoint(live);
    for await (const event of events) {
      options.onEvent?.(event);
      if (event.type === "turn_end") {
        live.push(event.message, ...event.toolResults);
        writeCheckpoint(live);
      }
    }
    const produced = await events.result();
    transcript.push(...produced);
    writeCheckpoint(transcript);
    const lastAssistant = produced
      .filter((m): m is AssistantMessage => m.role === "assistant")
      .at(-1);
    if (lastAssistant) {
      final = lastAssistant;
    }
    if (!final) {
      throw new Error("barbarian review produced no assistant message");
    }
    if (final.stopReason === "aborted") {
      throw new Error(`barbarian review aborted: ${final.errorMessage ?? "aborted"}`);
    }
    if (final.stopReason !== "error" || attempt >= MAX_CONTINUES) {
      break;
    }
    promptMessages = [{ role: "user", content: BARBARIAN_CONTINUE_NUDGE, timestamp: Date.now() }];
  }

  // The retry above recovers a SINGLE transient/content-filtered turn. If the
  // final turn is STILL an error after exhausting the retries, the review did
  // not complete — fail loudly. Do NOT fabricate a success: the CLI must exit
  // non-zero and the tool must surface an error rather than label a broken run
  // "complete".
  if (final.stopReason === "error") {
    throw new Error(
      `barbarian review failed after ${MAX_CONTINUES + 1} attempts: ${final.errorMessage ?? "error"}`,
    );
  }
  const findings = finalText(final).trimEnd();

  return {
    repo,
    source,
    target,
    artifactRoot,
    findings,
    ...(syntheticTarget !== undefined && { syntheticTarget }),
    modelSpec,
    reasoning,
  };
}
