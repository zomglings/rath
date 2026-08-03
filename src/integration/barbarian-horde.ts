/**
 * Deterministic horde integration: fake model, no API or key.
 *
 * Proves that concurrency zero retains the solo checkpoint contract, positive
 * concurrency bounds simultaneous workers, completed attacks can be steered
 * and reopened from their checkpoints, and a completed horde review resumes
 * without rerunning attacks.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@earendil-works/pi-ai/compat";
import {
  fauxAssistantMessage,
  fauxToolCall,
  registerFauxProvider,
} from "@earendil-works/pi-ai/compat";
import { type BarbarianResult, runBarbarianReview } from "../agents/barbarian.js";
import { barbarianBashSpawnHook, repairDanglingToolCalls } from "../agents/barbarian-horde.js";

const MODEL = "openai/gpt-4";
const WORKER_DELAY_MS = 180;

interface AttackView {
  id: string;
  status: string;
  result?: string;
}

interface RunMetrics {
  activeWorkers: number;
  maxActiveWorkers: number;
  workerCalls: number;
  nextToolCall: number;
}

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" }).trim();
}

function userText(context: Context): string {
  return context.messages
    .filter((message) => message.role === "user")
    .flatMap((message) =>
      typeof message.content === "string"
        ? [message.content]
        : message.content.filter((block) => block.type === "text").map((block) => block.text),
    )
    .join("\n");
}

function toolState(context: Context): { revision: number; attacks: Map<string, AttackView> } {
  let revision = 0;
  const attacks = new Map<string, AttackView>();
  for (const message of context.messages) {
    if (message.role !== "toolResult") continue;
    for (const block of message.content) {
      if (block.type !== "text") continue;
      let parsed: { revision?: number; attack?: AttackView; attacks?: AttackView[] };
      try {
        parsed = JSON.parse(block.text) as typeof parsed;
      } catch {
        continue;
      }
      revision = Math.max(revision, parsed.revision ?? 0);
      if (parsed.attack) {
        attacks.set(parsed.attack.id, parsed.attack);
      }
      for (const attack of parsed.attacks ?? []) {
        attacks.set(attack.id, attack);
      }
    }
  }
  return { revision, attacks };
}

function responseFactory(metrics: RunMetrics) {
  return async (context: Context) => {
    const systemPrompt = context.systemPrompt ?? "";
    if (systemPrompt.includes("Barbarian Horde attack agent")) {
      metrics.workerCalls++;
      metrics.activeWorkers++;
      metrics.maxActiveWorkers = Math.max(metrics.maxActiveWorkers, metrics.activeWorkers);
      try {
        await new Promise((resolveDelay) => setTimeout(resolveDelay, WORKER_DELAY_MS));
        const prompt = userText(context);
        const attackId = /ATTACK_ID: (attack-\d+)/.exec(prompt)?.[1] ?? "attack-unknown";
        const steered = prompt.includes("Chieftain steering");
        const verdict = attackId === "attack-001" && !steered ? "inconclusive" : "confirmed";
        return fauxAssistantMessage(
          `Attack: ${attackId}\nVerdict: ${verdict}\nFinding: scripted ${verdict}\nEvidence: deterministic faux worker\nReproduction: not staged\nSuggested fix: none`,
        );
      } finally {
        metrics.activeWorkers--;
      }
    }

    const { revision, attacks } = toolState(context);
    const toolCall = (name: string, arguments_: Record<string, unknown>) =>
      fauxToolCall(name, arguments_, { id: `faux-${metrics.nextToolCall++}` });
    if (attacks.size === 0) {
      return fauxAssistantMessage(
        [
          toolCall("launch_attack", {
            hypothesis: "first scripted hypothesis",
            objective: "obtain deterministic evidence",
          }),
          toolCall("launch_attack", {
            hypothesis: "second scripted hypothesis",
            objective: "obtain deterministic evidence",
          }),
          toolCall("launch_attack", {
            hypothesis: "open-ended coverage attack",
            objective: "find anything the first hypotheses missed",
          }),
        ],
        { stopReason: "toolUse" },
      );
    }
    const allTerminal = [...attacks.values()].every((attack) =>
      ["completed", "failed", "cancelled"].includes(attack.status),
    );
    const first = attacks.get("attack-001");
    if (allTerminal && first?.result?.includes("Verdict: inconclusive")) {
      return fauxAssistantMessage(
        toolCall("steer_attack", {
          attackId: "attack-001",
          instructions:
            "Reopen this checkpoint and convert the uncertainty into decisive evidence.",
        }),
        { stopReason: "toolUse" },
      );
    }
    if (allTerminal && first?.result?.includes("Verdict: confirmed")) {
      return fauxAssistantMessage(
        "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: deterministic horde exercise complete\n\nArtifacts:\n- attack checkpoints",
      );
    }
    return fauxAssistantMessage(toolCall("wait_for_attacks", { afterRevision: revision }), {
      stopReason: "toolUse",
    });
  };
}

function installResponses(
  registration: ReturnType<typeof registerFauxProvider>,
  metrics: RunMetrics,
): void {
  const factory = responseFactory(metrics);
  registration.setResponses(Array.from({ length: 200 }, () => factory));
}

function cleanupArtifact(repo: string, artifactRoot: string): void {
  const chieftainWorktree = join(artifactRoot, "chieftain-worktree");
  if (existsSync(chieftainWorktree)) {
    execFileSync("git", ["-C", repo, "worktree", "remove", "--force", chieftainWorktree], {
      stdio: "ignore",
    });
  }
  const attacks = join(artifactRoot, "attacks");
  if (existsSync(attacks)) {
    for (const id of readdirSync(attacks)) {
      const worktree = join(attacks, id, "worktree");
      if (existsSync(worktree)) {
        execFileSync("git", ["-C", repo, "worktree", "remove", "--force", worktree], {
          stdio: "ignore",
        });
      }
    }
  }
  rmSync(artifactRoot, { recursive: true, force: true });
}

function cleanupRun(repo: string, result: BarbarianResult): void {
  cleanupArtifact(repo, result.artifactRoot);
}

async function runCase(
  registration: ReturnType<typeof registerFauxProvider>,
  repo: string,
  concurrency: number,
): Promise<{ result: BarbarianResult; metrics: RunMetrics; elapsed: number }> {
  const metrics: RunMetrics = {
    activeWorkers: 0,
    maxActiveWorkers: 0,
    workerCalls: 0,
    nextToolCall: 1,
  };
  installResponses(registration, metrics);
  const started = performance.now();
  const result = await runBarbarianReview({
    repo,
    source: "HEAD~1",
    target: "HEAD",
    model: MODEL,
    reasoning: "off",
    concurrency,
  });
  return { result, metrics, elapsed: performance.now() - started };
}

async function main(): Promise<void> {
  const group = mkdtempSync(join(tmpdir(), "rath-barbarian-horde-test-"));
  const repo = join(group, "repo");
  execFileSync("git", ["init", "-b", "main", repo], { stdio: "ignore" });
  git(repo, "config", "user.email", "test@example.com");
  git(repo, "config", "user.name", "Test");
  writeFileSync(join(repo, "value.txt"), "one\n");
  git(repo, "add", "value.txt");
  git(repo, "commit", "-m", "initial");
  writeFileSync(join(repo, "value.txt"), "two\n");
  git(repo, "commit", "-am", "target");

  const registration = registerFauxProvider({
    api: "openai-responses",
    tokensPerSecond: 100_000,
  });
  let sequential: Awaited<ReturnType<typeof runCase>> | undefined;
  let parallel: Awaited<ReturnType<typeof runCase>> | undefined;
  let fastResult: BarbarianResult | undefined;
  let exclusiveResult: BarbarianResult | undefined;
  let cancelSuccessResult: BarbarianResult | undefined;
  let solo: BarbarianResult | undefined;
  try {
    registration.setResponses([
      fauxAssistantMessage(
        "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: solo dispatch exercise complete\n\nArtifacts:\n- none",
      ),
    ]);
    solo = await runBarbarianReview({
      repo,
      source: "HEAD~1",
      target: "HEAD",
      model: MODEL,
      reasoning: "off",
    });
    assert.equal(solo.concurrency, 0, "omitted concurrency preserves solo mode");
    assert.equal(
      JSON.parse(readFileSync(join(solo.artifactRoot, "checkpoint.json"), "utf8")).version,
      1,
      "solo mode preserves the version-1 checkpoint",
    );
    const repaired = repairDanglingToolCalls([
      { role: "user", content: "safe prefix", timestamp: 0 },
      fauxAssistantMessage(fauxToolCall("bash", { command: "true" }), {
        stopReason: "toolUse",
      }),
    ]);
    assert.equal(repaired.length, 3, "resume repair completes an interrupted tool batch");
    assert.equal(repaired.at(-1)?.role, "toolResult");
    const recoveredError = [
      { role: "user" as const, content: "start", timestamp: 0 },
      fauxAssistantMessage(fauxToolCall("bash", { command: "true" }), {
        stopReason: "error",
        errorMessage: "scripted",
      }),
      { role: "user" as const, content: "recover", timestamp: 1 },
      fauxAssistantMessage("done"),
    ];
    assert.equal(
      repairDanglingToolCalls(recoveredError).length,
      recoveredError.length,
      "errored tool-call turns do not truncate later recovery",
    );
    await assert.rejects(
      runBarbarianReview({
        repo,
        source: "HEAD~1",
        target: "HEAD",
        model: MODEL,
        reasoning: "off",
        concurrency: 1,
        hordeModel: "not-a-provider/not-a-model",
      }),
      /Unknown model/,
      "invalid horde models fail during preflight",
    );

    const expectedSource = git(repo, "rev-parse", "HEAD~1");
    const alien = join(group, "alien");
    execFileSync("git", ["init", "-b", "main", alien], { stdio: "ignore" });
    git(alien, "config", "user.email", "test@example.com");
    git(alien, "config", "user.name", "Test");
    writeFileSync(join(alien, "alien.txt"), "alien\n");
    git(alien, "add", "alien.txt");
    git(alien, "commit", "-m", "alien");
    const hookMarker = join(group, "post-checkout-hook-ran");
    const postCheckoutHook = join(repo, ".git", "hooks", "post-checkout");
    writeFileSync(postCheckoutHook, `#!/bin/sh\nprintf ran > "${hookMarker}"\nexit 97\n`);
    chmodSync(postCheckoutHook, 0o755);
    const previousGitDir = process.env.GIT_DIR;
    process.env.GIT_DIR = join(alien, ".git");
    try {
      const sanitized = barbarianBashSpawnHook(join(group, "bash-artifacts"))({
        command: "git status",
        cwd: repo,
        env: {
          ...process.env,
          GIT_WORK_TREE: alien,
          GIT_INDEX_FILE: join(alien, ".git", "index"),
        },
      });
      assert.equal(sanitized.env.GIT_DIR, undefined);
      assert.equal(sanitized.env.GIT_WORK_TREE, undefined);
      assert.equal(sanitized.env.GIT_INDEX_FILE, undefined);
      assert.equal(sanitized.env.GIT_CONFIG_KEY_0, "core.hooksPath");
      assert.ok(sanitized.env.GIT_CONFIG_VALUE_0?.endsWith(".empty-git-hooks"));
      sequential = await runCase(registration, repo, 1);
    } finally {
      if (previousGitDir === undefined) {
        delete process.env.GIT_DIR;
      } else {
        process.env.GIT_DIR = previousGitDir;
      }
    }
    assert.equal(sequential.result.source, expectedSource, "inherited GIT_DIR is ignored");
    assert.equal(existsSync(hookMarker), false, "horde worktree creation disables hooks");
    parallel = await runCase(registration, repo, 2);

    assert.equal(sequential.metrics.maxActiveWorkers, 1, "concurrency 1 runs one worker");
    assert.equal(parallel.metrics.maxActiveWorkers, 2, "concurrency 2 runs two workers");
    assert.equal(sequential.metrics.workerCalls, 4, "steering reopens one completed attack");
    assert.equal(parallel.metrics.workerCalls, 4, "parallel run executes the same attack work");
    assert.ok(
      parallel.elapsed < sequential.elapsed - 100,
      `parallel wall time should improve (${parallel.elapsed.toFixed(0)}ms vs ${sequential.elapsed.toFixed(0)}ms)`,
    );

    const fastFinding = "VERIFIED FAST ATTACK FINDING";
    const fastFactory = async (context: Context) => {
      if ((context.systemPrompt ?? "").includes("Barbarian Horde attack agent")) {
        return fauxAssistantMessage(
          `Attack: attack-001\nVerdict: confirmed\nFinding: ${fastFinding}\nEvidence: deterministic\nReproduction: none\nSuggested fix: none`,
        );
      }
      const { attacks } = toolState(context);
      if (attacks.size === 0) {
        return fauxAssistantMessage(
          fauxToolCall(
            "launch_attack",
            { hypothesis: "finish before polling", objective: "test terminal snapshot delivery" },
            { id: "fast-launch" },
          ),
          { stopReason: "toolUse" },
        );
      }
      if (
        [...attacks.values()].some((attack) => attack.result?.includes(fastFinding)) ||
        userText(context).includes(fastFinding)
      ) {
        return fauxAssistantMessage(
          `Reviewed: scripted-source..scripted-target\n\nFindings:\n\n1. minor: ${fastFinding}\n\nArtifacts:\n- attack checkpoint`,
        );
      }
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 80));
      return fauxAssistantMessage(
        "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: premature synthesis\n\nArtifacts:\n- none",
      );
    };
    registration.setResponses(Array.from({ length: 30 }, () => fastFactory));
    fastResult = await runBarbarianReview({
      repo,
      source: "HEAD~1",
      target: "HEAD",
      model: MODEL,
      reasoning: "off",
      concurrency: 1,
    });
    assert.match(
      fastResult.findings,
      new RegExp(fastFinding),
      "a terminal snapshot is delivered when an attack finishes before follow-up polling",
    );

    const exclusiveMetrics: RunMetrics = {
      activeWorkers: 0,
      maxActiveWorkers: 0,
      workerCalls: 0,
      nextToolCall: 1,
    };
    const exclusiveFactory = async (context: Context) => {
      if ((context.systemPrompt ?? "").includes("Barbarian Horde attack agent")) {
        exclusiveMetrics.workerCalls++;
        exclusiveMetrics.activeWorkers++;
        exclusiveMetrics.maxActiveWorkers = Math.max(
          exclusiveMetrics.maxActiveWorkers,
          exclusiveMetrics.activeWorkers,
        );
        try {
          await new Promise((resolveDelay) => setTimeout(resolveDelay, 300));
          return fauxAssistantMessage(
            "Attack: attack-001\nVerdict: confirmed\nFinding: exclusive owner\nEvidence: deterministic\nReproduction: none\nSuggested fix: none",
          );
        } finally {
          exclusiveMetrics.activeWorkers--;
        }
      }
      const { revision, attacks } = toolState(context);
      if (attacks.size === 0) {
        return fauxAssistantMessage(
          fauxToolCall(
            "launch_attack",
            { hypothesis: "exclusive review", objective: "hold the artifact lock" },
            { id: "exclusive-launch" },
          ),
          { stopReason: "toolUse" },
        );
      }
      if ([...attacks.values()].some((attack) => attack.status === "completed")) {
        return fauxAssistantMessage(
          "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: exclusive run complete\n\nArtifacts:\n- attack checkpoint",
        );
      }
      return fauxAssistantMessage(
        fauxToolCall(
          "wait_for_attacks",
          { afterRevision: revision },
          {
            id: `exclusive-wait-${exclusiveMetrics.nextToolCall++}`,
          },
        ),
        { stopReason: "toolUse" },
      );
    };
    registration.setResponses(Array.from({ length: 50 }, () => exclusiveFactory));
    let announceExclusiveRoot: ((root: string) => void) | undefined;
    const exclusiveRootReady = new Promise<string>((resolveRoot) => {
      announceExclusiveRoot = resolveRoot;
    });
    const exclusiveRun = runBarbarianReview({
      repo,
      source: "HEAD~1",
      target: "HEAD",
      model: MODEL,
      reasoning: "off",
      concurrency: 1,
      onArtifactRoot: (root) => announceExclusiveRoot?.(root),
    });
    const exclusiveRoot = await exclusiveRootReady;
    const workerDeadline = Date.now() + 2_000;
    while (exclusiveMetrics.activeWorkers === 0 && Date.now() < workerDeadline) {
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 5));
    }
    assert.equal(exclusiveMetrics.activeWorkers, 1, "first review owns a live worker");
    await assert.rejects(
      runBarbarianReview({
        resume: exclusiveRoot,
        concurrency: 1,
        model: MODEL,
        reasoning: "off",
      }),
      /horde review is already active/,
      "a concurrent resume is rejected while the artifact owner is alive",
    );
    exclusiveResult = await exclusiveRun;
    assert.equal(
      exclusiveMetrics.maxActiveWorkers,
      1,
      "concurrent resume cannot duplicate workers",
    );

    const checkpointPath = join(parallel.result.artifactRoot, "checkpoint.json");
    const attackCheckpointPath = join(
      parallel.result.artifactRoot,
      "attacks",
      "attack-001",
      "checkpoint.json",
    );
    assert.equal(JSON.parse(readFileSync(checkpointPath, "utf8")).version, 2);
    const attackCheckpoint = JSON.parse(readFileSync(attackCheckpointPath, "utf8")) as {
      status: string;
      steering: { delivered: boolean }[];
    };
    assert.equal(attackCheckpoint.status, "completed");
    assert.ok(attackCheckpoint.steering.some((instruction) => instruction.delivered));

    await assert.rejects(
      runBarbarianReview({ resume: parallel.result.artifactRoot, concurrency: 0 }),
      /requires --concurrency <positive integer>/,
    );
    await assert.rejects(
      runBarbarianReview({
        resume: join(parallel.result.artifactRoot, "attacks", "attack-001"),
      }),
      /attack checkpoint must be resumed through its parent horde artifact root/,
    );
    const workerCallsBeforeResume = parallel.metrics.workerCalls;
    const staleLock = join(parallel.result.artifactRoot, ".review-lock");
    mkdirSync(staleLock);
    writeFileSync(
      join(staleLock, "owner.json"),
      JSON.stringify({
        version: 1,
        hostname: hostname(),
        pid: 2_147_483_647,
        token: "stale-owner",
        startedAt: "2000-01-01T00:00:00.000Z",
      }),
    );
    installResponses(registration, parallel.metrics);
    const resumed = await runBarbarianReview({
      resume: parallel.result.artifactRoot,
      concurrency: 2,
      model: MODEL,
      reasoning: "off",
    });
    assert.equal(resumed.concurrency, 2);
    assert.equal(
      parallel.metrics.workerCalls,
      workerCallsBeforeResume,
      "completed attacks are not rerun on resume",
    );
    assert.equal(
      existsSync(staleLock),
      false,
      "dead local review locks are recovered and released",
    );

    const checkpointPathForBootstrap = join(parallel.result.artifactRoot, "checkpoint.json");
    const bootstrapCheckpoint = JSON.parse(readFileSync(checkpointPathForBootstrap, "utf8")) as {
      source: string;
      chieftainMessages: unknown[];
      instructions?: string;
    };
    bootstrapCheckpoint.chieftainMessages = [];
    bootstrapCheckpoint.instructions = "RESUME-MUST-RETAIN-THIS-INSTRUCTION";
    writeFileSync(checkpointPathForBootstrap, JSON.stringify(bootstrapCheckpoint, null, 2));
    let resumeBootstrap = "";
    registration.setResponses([
      async (context) => {
        resumeBootstrap = userText(context);
        return fauxAssistantMessage(
          "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: bootstrap restored\n\nArtifacts:\n- existing attack checkpoints",
        );
      },
    ]);
    await runBarbarianReview({
      resume: parallel.result.artifactRoot,
      concurrency: 2,
      model: MODEL,
      reasoning: "off",
    });
    assert.ok(resumeBootstrap.includes(`SOURCE: ${bootstrapCheckpoint.source}`));
    assert.ok(resumeBootstrap.includes("RESUME-MUST-RETAIN-THIS-INSTRUCTION"));

    const cancelSuccessMetrics: RunMetrics = {
      activeWorkers: 0,
      maxActiveWorkers: 0,
      workerCalls: 0,
      nextToolCall: 1,
    };
    const cancelSuccessFactory = async (context: Context) => {
      if ((context.systemPrompt ?? "").includes("Barbarian Horde attack agent")) {
        cancelSuccessMetrics.workerCalls++;
        cancelSuccessMetrics.activeWorkers++;
        cancelSuccessMetrics.maxActiveWorkers = Math.max(
          cancelSuccessMetrics.maxActiveWorkers,
          cancelSuccessMetrics.activeWorkers,
        );
        try {
          const attackId = /ATTACK_ID: (attack-\d+)/.exec(userText(context))?.[1];
          await new Promise((resolveDelay) =>
            setTimeout(resolveDelay, attackId === "attack-001" ? 30 : 450),
          );
          return fauxAssistantMessage(
            `Attack: ${attackId}\nVerdict: confirmed\nFinding: scripted result\nEvidence: deterministic\nReproduction: none\nSuggested fix: none`,
          );
        } finally {
          cancelSuccessMetrics.activeWorkers--;
        }
      }
      const { revision, attacks } = toolState(context);
      const call = (name: string, arguments_: Record<string, unknown>) =>
        fauxToolCall(name, arguments_, {
          id: `cancel-success-${cancelSuccessMetrics.nextToolCall++}`,
        });
      if (attacks.size === 0) {
        return fauxAssistantMessage(
          [
            call("launch_attack", { hypothesis: "quick success", objective: "complete" }),
            call("launch_attack", { hypothesis: "slow cancellation", objective: "be cancelled" }),
          ],
          { stopReason: "toolUse" },
        );
      }
      const first = attacks.get("attack-001");
      const second = attacks.get("attack-002");
      if (first?.status === "completed" && second?.status === "cancelled") {
        return fauxAssistantMessage(
          "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: cancellation settled\n\nArtifacts:\n- attack checkpoints",
        );
      }
      if (first?.status === "completed" && second?.status === "running") {
        return fauxAssistantMessage(call("cancel_attack", { attackId: "attack-002" }), {
          stopReason: "toolUse",
        });
      }
      return fauxAssistantMessage(call("wait_for_attacks", { afterRevision: revision }), {
        stopReason: "toolUse",
      });
    };
    registration.setResponses(Array.from({ length: 100 }, () => cancelSuccessFactory));
    cancelSuccessResult = await runBarbarianReview({
      repo,
      source: "HEAD~1",
      target: "HEAD",
      model: MODEL,
      reasoning: "off",
      concurrency: 2,
    });
    assert.equal(
      cancelSuccessMetrics.activeWorkers,
      0,
      "successful review waits for cancelled worker settlement",
    );

    const failureMetrics: RunMetrics = {
      activeWorkers: 0,
      maxActiveWorkers: 0,
      workerCalls: 0,
      nextToolCall: 1,
    };
    const failureFactory = async (context: Context) => {
      if ((context.systemPrompt ?? "").includes("Barbarian Horde attack agent")) {
        failureMetrics.workerCalls++;
        failureMetrics.activeWorkers++;
        try {
          await new Promise((resolveDelay) => setTimeout(resolveDelay, 500));
          return fauxAssistantMessage(
            "Attack: attack-001\nVerdict: confirmed\nFinding: too late\nEvidence: scripted\nReproduction: not staged\nSuggested fix: none",
          );
        } finally {
          failureMetrics.activeWorkers--;
        }
      }
      const { attacks } = toolState(context);
      if (attacks.size === 0) {
        return fauxAssistantMessage(
          fauxToolCall(
            "launch_attack",
            { hypothesis: "slow attack", objective: "remain live during chieftain failure" },
            { id: `failure-${failureMetrics.nextToolCall++}` },
          ),
          { stopReason: "toolUse" },
        );
      }
      return fauxAssistantMessage("scripted chieftain failure", {
        stopReason: "error",
        errorMessage: "scripted chieftain failure",
      });
    };
    registration.setResponses(Array.from({ length: 50 }, () => failureFactory));
    let failedArtifactRoot: string | undefined;
    await assert.rejects(
      runBarbarianReview({
        repo,
        source: "HEAD~1",
        target: "HEAD",
        model: MODEL,
        reasoning: "off",
        concurrency: 1,
        onArtifactRoot: (root) => {
          failedArtifactRoot = root;
        },
      }),
      /agent failed after 3 attempts/,
    );
    assert.equal(failureMetrics.activeWorkers, 0, "failure waits for worker settlement");
    assert.ok(failedArtifactRoot, "failed review exposes its artifact root");
    const failedAttack = JSON.parse(
      readFileSync(join(failedArtifactRoot!, "attacks", "attack-001", "checkpoint.json"), "utf8"),
    ) as { status: string };
    assert.equal(failedAttack.status, "interrupted", "live worker is checkpointed as interrupted");
    cleanupArtifact(repo, failedArtifactRoot!);

    const cancelMetrics: RunMetrics = {
      activeWorkers: 0,
      maxActiveWorkers: 0,
      workerCalls: 0,
      nextToolCall: 1,
    };
    const cancelFactory = async (context: Context) => {
      if ((context.systemPrompt ?? "").includes("Barbarian Horde attack agent")) {
        cancelMetrics.workerCalls++;
        return fauxAssistantMessage(
          "Attack: attack-001\nVerdict: confirmed\nFinding: should not run\nEvidence: none\nReproduction: none\nSuggested fix: none",
        );
      }
      const { attacks } = toolState(context);
      if (attacks.size === 0) {
        return fauxAssistantMessage(
          [
            fauxToolCall(
              "launch_attack",
              { hypothesis: "cancel immediately", objective: "must not start" },
              { id: "cancel-launch" },
            ),
            fauxToolCall("cancel_attack", { attackId: "attack-001" }, { id: "cancel-now" }),
          ],
          { stopReason: "toolUse" },
        );
      }
      return fauxAssistantMessage(
        "Reviewed: scripted-source..scripted-target\n\nFindings:\n\nNo finding: cancelled\n\nArtifacts:\n- none",
      );
    };
    registration.setResponses(Array.from({ length: 20 }, () => cancelFactory));
    let cancelledArtifactRoot: string | undefined;
    await assert.rejects(
      runBarbarianReview({
        repo,
        source: "HEAD~1",
        target: "HEAD",
        model: MODEL,
        reasoning: "off",
        concurrency: 1,
        onArtifactRoot: (root) => {
          cancelledArtifactRoot = root;
        },
      }),
      /without a successful attack/,
    );
    assert.equal(cancelMetrics.workerCalls, 0, "same-batch cancellation prevents worker start");
    assert.ok(cancelledArtifactRoot);
    assert.equal(
      existsSync(join(cancelledArtifactRoot!, "attacks", "attack-001", "worktree")),
      false,
      "same-batch cancellation prevents worktree creation",
    );
    cleanupArtifact(repo, cancelledArtifactRoot!);

    log(
      `Concurrency bound and speedup verified: 1=${sequential.elapsed.toFixed(0)}ms, ` +
        `2=${parallel.elapsed.toFixed(0)}ms`,
    );
    log("Omitted concurrency preserved the solo reviewer and checkpoint format.");
    log("Steering checkpoint and resume reuse verified.");
    log("Fast terminal attack snapshots reached the chieftain before synthesis.");
    log("Exclusive artifact ownership rejected concurrent resume and recovered a dead owner.");
    log("Empty bootstrap and interrupted tool batches recover safely.");
    log("Inherited repository-selection Git environment was ignored by setup and bash tools.");
    log("Synthetic targets and horde Git commands disabled repository hooks.");
    log("Same-batch cancellation prevented worker startup.");
    log("Successful review awaited cancelled worker settlement.");
    log("Chieftain failure interrupted and awaited live workers.");
  } finally {
    if (solo) cleanupRun(repo, solo);
    if (sequential) cleanupRun(repo, sequential.result);
    if (parallel) cleanupRun(repo, parallel.result);
    if (fastResult) cleanupRun(repo, fastResult);
    if (exclusiveResult) cleanupRun(repo, exclusiveResult);
    if (cancelSuccessResult) cleanupRun(repo, cancelSuccessResult);
    registration.unregister();
    rmSync(group, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? (error.stack ?? error.message) : error);
  process.exit(1);
});
