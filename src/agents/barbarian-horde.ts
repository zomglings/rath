import { execFileSync } from "node:child_process";
import { existsSync, lstatSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  type AgentContext,
  type AgentEvent,
  type AgentLoopConfig,
  type AgentMessage,
  type AgentTool,
  type AgentToolResult,
  agentLoop,
  type StreamFn,
} from "@earendil-works/pi-agent-core";
import {
  type AssistantMessage,
  type Message,
  type SimpleStreamOptions,
  streamSimple,
  type Usage,
} from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import {
  acquireBarbarianRunLock,
  canonicalBarbarianArtifactRoot,
  createLockedBarbarianArtifactRoot,
} from "../barbarian-artifacts.js";
import { hordeIntelligence } from "../barbarian-defaults.js";
import { REASONING_LEVELS, type ReasoningLevel, resolveModel } from "../models.js";
import {
  type BarbarianOptions,
  type BarbarianResult,
  barbarianGitEnvironment,
  barbarianHooksPath,
  barbarianWorktreeAddArgs,
  createSyntheticTarget,
  hasChanges,
  repoRoot,
  resolveSource,
  totalUsage,
} from "./barbarian.js";

const REVIEW_CHECKPOINT = "checkpoint.json";
const ATTACK_CHECKPOINT = "checkpoint.json";
const MAX_CONTINUES = 2;
const WAIT_TIMEOUT_MS = 20_000;

const CHIEFTAIN_SYSTEM_PROMPT = `You are the Barbarian Chieftain. You run non-interactively inside a git repository and command a horde of independent attack agents. Your task is to review SOURCE..TARGET and produce the final findings report. Never ask questions or request approval.

Never modify the user's working tree. You run in a dedicated detached worktree for TARGET; delegate every reproduction and intentional code change to an attack agent's separate worktree.

Inspect the diff and surrounding contracts yourself, then generate concrete defect hypotheses. Dispatch independent hypotheses with launch_attack. Issue multiple launch_attack calls in the same turn whenever possible; they run concurrently up to the configured limit. Include at least one open-ended coverage attack whose job is to find defects outside your initial hypothesis set.

Each attack has its own isolated worktree and durable checkpoint. Use wait_for_attacks to observe progress, steer_attack to redirect or deepen an investigation, and cancel_attack to abandon work that no longer matters. A completed or failed attack can be reopened with steer_attack. Do not trust worker conclusions: compare their evidence to the code, reject weak claims, deduplicate overlapping findings, and request stronger reproduction when needed.

Do not finish while attacks are queued or running. Report only issues with a plausible path to user-visible failure, production risk, broken tests, or significant future maintenance cost. Ignore style-only and coverage-only complaints.

Your final assistant message IS the findings report. It must contain the report only:

Reviewed: <source-sha>..<target-sha>

Findings:

1. <blocker|major|minor>: <plain defect statement>
   Evidence: <file:line/function, command output, or reasoning chain>
   Reproduction: <exact commands and attack artifact paths, or "not staged">
   Fix: <specific required change>

No finding: <only if no findings survived verification>

Artifacts:
- <attack artifact paths, or "none">

Be terse, skeptical, and evidence-driven.`;

const ATTACK_SYSTEM_PROMPT = `You are a Barbarian Horde attack agent. You receive one hypothesis or bounded investigation from a chieftain. Prove or refute it in an isolated git worktree. Never ask the human questions and never modify the user's working tree.

Read the relevant diff and surrounding code. Construct the smallest useful attack vector or reproduction. Run exact commands when practical. Treat the hypothesis as untrusted: rejection with decisive evidence is as useful as confirmation. Steering from the chieftain may arrive between turns; follow the newest instruction.

Your final assistant message must contain only this result:

Attack: <attack-id>
Verdict: <confirmed|rejected|inconclusive>
Finding: <defect and impact, or "none">
Evidence: <specific code paths and observed output>
Reproduction: <exact commands and artifact paths, or "not staged">
Suggested fix: <specific change, or "none">

Do not inflate severity, report style nits, or claim proof you did not obtain.`;

const CONTINUE_NUDGE =
  "The previous turn ended in a provider error. Continue neutrally from the surviving " +
  "transcript and finish the assigned work.";
const ATTACK_RESUME_NUDGE =
  "Resume this attack from its durable checkpoint. Revalidate any interrupted command, " +
  "finish the investigation, and return the required attack result.";
const CHIEFTAIN_RESUME_NUDGE =
  "Resume this horde review from its durable checkpoint. Inspect the restored attack states, " +
  "continue or steer them as needed, and finish with the complete findings report.";

export type AttackStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

interface SteeringInstruction {
  sequence: number;
  content: string;
  delivered: boolean;
}

interface AttackCheckpoint {
  version: 1;
  kind: "horde-attack";
  id: string;
  hypothesis: string;
  objective: string;
  repo: string;
  source: string;
  target: string;
  artifactRoot: string;
  worktree: string;
  modelSpec: string;
  reasoning: ReasoningLevel;
  status: AttackStatus;
  messages: AgentMessage[];
  steering: SteeringInstruction[];
  nextSteeringSequence: number;
  latestProgress?: string;
  result?: string;
  error?: string;
}

interface HordeCheckpoint {
  version: 2;
  mode: "horde";
  repo: string;
  source: string;
  target: string;
  syntheticTarget?: string;
  artifactRoot: string;
  chieftainWorktree: string;
  chieftainModelSpec: string;
  chieftainReasoning: ReasoningLevel;
  hordeModelSpec: string;
  hordeReasoning: ReasoningLevel;
  concurrency: number;
  instructions?: string;
  chieftainMessages: AgentMessage[];
  attackIds: string[];
  nextAttackNumber: number;
  revision: number;
}

export interface AttackSnapshot {
  id: string;
  hypothesis: string;
  objective: string;
  status: AttackStatus;
  artifactRoot: string;
  latestProgress?: string;
  result?: string;
  error?: string;
}

function atomicWriteJson(path: string, value: unknown): void {
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(value, null, 2));
  renameSync(temporary, path);
}

function readJson<T>(path: string): T {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as T;
  } catch (error) {
    throw new Error(
      `${path} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function git(repo: string, args: string[]): string {
  try {
    return execFileSync("git", ["-C", repo, ...args], {
      encoding: "utf8",
      maxBuffer: 1024 * 1024 * 1024,
      env: barbarianGitEnvironment({
        GIT_AUTHOR_NAME: process.env.GIT_AUTHOR_NAME ?? "Barbarian Reviewer",
        GIT_AUTHOR_EMAIL: process.env.GIT_AUTHOR_EMAIL ?? "barbarian@rath.invalid",
        GIT_COMMITTER_NAME: process.env.GIT_COMMITTER_NAME ?? "Barbarian Reviewer",
        GIT_COMMITTER_EMAIL: process.env.GIT_COMMITTER_EMAIL ?? "barbarian@rath.invalid",
      }),
    }).trim();
  } catch (error) {
    throw new Error(
      `git -C ${repo} ${args.join(" ")} failed: ${error instanceof Error ? error.message : error}`,
    );
  }
}

function resolveCommit(repo: string, value: string): string {
  return git(repo, ["rev-parse", "--verify", `${value}^{commit}`]);
}

function finalText(message: AssistantMessage): string {
  return message.content
    .filter((block): block is { type: "text"; text: string } => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trimEnd();
}

function progressText(message: AssistantMessage): string | undefined {
  const text = finalText(message).trim();
  if (text) {
    return text.length > 1200 ? `${text.slice(0, 1200)}…` : text;
  }
  const calls = message.content
    .filter((block) => block.type === "toolCall")
    .map((block) => block.name);
  return calls.length > 0 ? `Tools: ${calls.join(", ")}` : undefined;
}

function convertToLlm(messages: AgentMessage[]): Message[] {
  const droppedToolCallIds = new Set<string>();
  for (const message of messages) {
    if (
      message.role === "assistant" &&
      (message.stopReason === "error" || message.stopReason === "aborted")
    ) {
      for (const block of message.content) {
        if (block.type === "toolCall") {
          droppedToolCallIds.add(block.id);
        }
      }
    }
  }
  return messages.filter((message): message is Message => {
    if (message.role === "user") {
      return true;
    }
    if (message.role === "toolResult") {
      return !droppedToolCallIds.has(message.toolCallId);
    }
    return (
      message.role === "assistant" &&
      message.stopReason !== "error" &&
      message.stopReason !== "aborted"
    );
  });
}

/** Complete an interrupted tool batch so a restored transcript remains provider-valid. */
export function repairDanglingToolCalls(messages: AgentMessage[]): AgentMessage[] {
  const repaired = [...messages];
  for (let index = 0; index < repaired.length; index++) {
    const message = repaired[index];
    if (message?.role !== "assistant") continue;
    if (message.stopReason === "error" || message.stopReason === "aborted") continue;
    const calls = message.content.filter((block) => block.type === "toolCall");
    const missing = new Map(calls.map((block) => [block.id, block.name]));
    if (missing.size === 0) continue;
    let cursor = index + 1;
    while (cursor < repaired.length && repaired[cursor]?.role === "toolResult") {
      const result = repaired[cursor];
      if (result?.role === "toolResult") {
        missing.delete(result.toolCallId);
      }
      cursor++;
    }
    if (missing.size > 0) {
      const synthetic: AgentMessage[] = [...missing].map(([toolCallId, toolName]) => ({
        role: "toolResult" as const,
        toolCallId,
        toolName,
        content: [
          {
            type: "text" as const,
            text: "Tool execution was interrupted before a durable result was checkpointed.",
          },
        ],
        isError: true,
        timestamp: Date.now(),
      }));
      repaired.splice(cursor, 0, ...synthetic);
      index = cursor + synthetic.length - 1;
    }
  }
  return repaired;
}

const streamWithoutWebSearch: StreamFn = (model, context, options) =>
  streamSimple(model, context, { ...options, webSearch: false } as SimpleStreamOptions);

interface BashSpawnContext {
  command: string;
  cwd: string;
  env: NodeJS.ProcessEnv;
}

export function barbarianBashSpawnHook(
  artifactRoot: string,
): (context: BashSpawnContext) => BashSpawnContext {
  const hooks = barbarianHooksPath(artifactRoot);
  return (context) => {
    const env = barbarianGitEnvironment(context.env);
    const parsedCount = env.GIT_CONFIG_COUNT === undefined ? 0 : Number(env.GIT_CONFIG_COUNT);
    const count = Number.isSafeInteger(parsedCount) && parsedCount >= 0 ? parsedCount : 0;
    env.GIT_CONFIG_COUNT = String(count + 1);
    env[`GIT_CONFIG_KEY_${count}`] = "core.hooksPath";
    env[`GIT_CONFIG_VALUE_${count}`] = hooks;
    return { ...context, env };
  };
}

interface AgentRunOptions {
  context: AgentContext;
  config: AgentLoopConfig;
  transcript: AgentMessage[];
  promptMessages: AgentMessage[];
  signal?: AbortSignal;
  onEvent?: (event: AgentEvent) => void;
  onMessages: (messages: AgentMessage[]) => void;
}

async function runAgentWithRecovery(options: AgentRunOptions): Promise<{
  messages: AgentMessage[];
  final: AssistantMessage;
}> {
  const transcript = [...options.transcript];
  let promptMessages = options.promptMessages;
  let final: AssistantMessage | undefined;

  for (let attempt = 0; ; attempt++) {
    const live = [...transcript];
    const events = agentLoop(
      promptMessages,
      { ...options.context, messages: transcript },
      options.config,
      options.signal,
      streamWithoutWebSearch,
    );
    for await (const event of events) {
      options.onEvent?.(event);
      if (event.type === "message_end") {
        live.push(event.message);
        options.onMessages(live);
      } else if (event.type === "turn_end") {
        options.onMessages(live);
      }
    }
    const produced = await events.result();
    transcript.push(...produced);
    options.onMessages(transcript);
    const last = produced
      .filter((message): message is AssistantMessage => message.role === "assistant")
      .at(-1);
    if (last) {
      final = last;
    }
    if (!final) {
      throw new Error("agent produced no assistant message");
    }
    if (final.stopReason === "aborted") {
      throw new Error(`agent aborted: ${final.errorMessage ?? "aborted"}`);
    }
    if (final.stopReason !== "error" || attempt >= MAX_CONTINUES) {
      break;
    }
    promptMessages = [{ role: "user", content: CONTINUE_NUDGE, timestamp: Date.now() }];
  }

  if (final.stopReason === "error") {
    throw new Error(
      `agent failed after ${MAX_CONTINUES + 1} attempts: ${final.errorMessage ?? "error"}`,
    );
  }
  return { messages: transcript, final };
}

interface CoordinatorOptions {
  checkpoint: HordeCheckpoint;
  records: Map<string, AttackCheckpoint>;
  signal?: AbortSignal;
  onEvent?: (attackId: string, event: AgentEvent) => void;
}

class HordeCoordinator {
  private readonly checkpoint: HordeCheckpoint;
  private readonly records: Map<string, AttackCheckpoint>;
  private readonly signal?: AbortSignal;
  private readonly onEvent?: (attackId: string, event: AgentEvent) => void;
  private readonly queue: string[] = [];
  private readonly controllers = new Map<string, AbortController>();
  private readonly settlements = new Map<string, Promise<void>>();
  private readonly listeners = new Set<() => void>();
  private active = 0;
  private stopping = false;
  private schedulePending = false;

  constructor(options: CoordinatorOptions) {
    this.checkpoint = options.checkpoint;
    this.records = options.records;
    this.signal = options.signal;
    this.onEvent = options.onEvent;
    for (const record of this.records.values()) {
      record.modelSpec = this.checkpoint.hordeModelSpec;
      record.reasoning = this.checkpoint.hordeReasoning;
      if (
        record.status === "queued" ||
        record.status === "running" ||
        record.status === "interrupted"
      ) {
        record.status = "queued";
        record.error = undefined;
        this.saveAttack(record);
        this.queue.push(record.id);
      }
    }
    if (this.signal?.aborted) {
      this.beginShutdown("review interrupted");
    } else {
      this.signal?.addEventListener("abort", () => this.beginShutdown("review interrupted"), {
        once: true,
      });
      this.requestSchedule();
    }
  }

  get revision(): number {
    return this.checkpoint.revision;
  }

  private saveReview(): void {
    atomicWriteJson(join(this.checkpoint.artifactRoot, REVIEW_CHECKPOINT), this.checkpoint);
  }

  private saveAttack(record: AttackCheckpoint): void {
    atomicWriteJson(join(record.artifactRoot, ATTACK_CHECKPOINT), record);
  }

  private changed(): void {
    this.checkpoint.revision++;
    this.saveReview();
    for (const listener of this.listeners) {
      listener();
    }
    this.listeners.clear();
  }

  private snapshot(record: AttackCheckpoint): AttackSnapshot {
    return {
      id: record.id,
      hypothesis: record.hypothesis,
      objective: record.objective,
      status: record.status,
      artifactRoot: record.artifactRoot,
      ...(record.latestProgress !== undefined && { latestProgress: record.latestProgress }),
      ...(record.result !== undefined && { result: record.result }),
      ...(record.error !== undefined && { error: record.error }),
    };
  }

  snapshots(): AttackSnapshot[] {
    return this.checkpoint.attackIds
      .map((id) => this.records.get(id))
      .filter((record): record is AttackCheckpoint => record !== undefined)
      .map((record) => this.snapshot(record));
  }

  hasNonTerminal(): boolean {
    return [...this.records.values()].some(
      (record) => record.status === "queued" || record.status === "running",
    );
  }

  messages(): AgentMessage[] {
    return [...this.records.values()].flatMap((record) => record.messages);
  }

  launch(hypothesis: string, objective: string): AttackSnapshot {
    const number = this.checkpoint.nextAttackNumber++;
    const id = `attack-${String(number).padStart(3, "0")}`;
    const artifactRoot = join(this.checkpoint.artifactRoot, "attacks", id);
    assertCheckpointPathComponents(
      this.checkpoint.artifactRoot,
      join(this.checkpoint.artifactRoot, "attacks"),
      "checkpoint attacks directory",
    );
    assertCheckpointPathComponents(
      this.checkpoint.artifactRoot,
      artifactRoot,
      `attack ${id} artifact root`,
    );
    mkdirSync(artifactRoot, { recursive: true });
    assertCheckpointPathComponents(
      this.checkpoint.artifactRoot,
      artifactRoot,
      `attack ${id} artifact root`,
    );
    const record: AttackCheckpoint = {
      version: 1,
      kind: "horde-attack",
      id,
      hypothesis,
      objective,
      repo: this.checkpoint.repo,
      source: this.checkpoint.source,
      target: this.checkpoint.target,
      artifactRoot,
      worktree: join(artifactRoot, "worktree"),
      modelSpec: this.checkpoint.hordeModelSpec,
      reasoning: this.checkpoint.hordeReasoning,
      status: "queued",
      messages: [],
      steering: [],
      nextSteeringSequence: 1,
    };
    this.records.set(id, record);
    this.checkpoint.attackIds.push(id);
    this.queue.push(id);
    this.saveAttack(record);
    this.changed();
    this.requestSchedule();
    return this.snapshot(record);
  }

  steer(id: string, content: string): AttackSnapshot {
    const record = this.records.get(id);
    if (!record) {
      throw new Error(`unknown attack: ${id}`);
    }
    if (record.status === "cancelled") {
      throw new Error(`cannot steer cancelled attack: ${id}`);
    }
    record.steering.push({
      sequence: record.nextSteeringSequence++,
      content,
      delivered: false,
    });
    if (
      record.status === "completed" ||
      record.status === "failed" ||
      record.status === "interrupted"
    ) {
      record.status = "queued";
      record.result = undefined;
      record.error = undefined;
      this.queue.push(id);
    }
    this.saveAttack(record);
    this.changed();
    this.requestSchedule();
    return this.snapshot(record);
  }

  cancel(id: string): AttackSnapshot {
    const record = this.records.get(id);
    if (!record) {
      throw new Error(`unknown attack: ${id}`);
    }
    if (record.status === "completed" || record.status === "failed") {
      throw new Error(`attack is already terminal: ${id}`);
    }
    record.status = "cancelled";
    record.error = "cancelled by chieftain";
    this.controllers.get(id)?.abort();
    this.saveAttack(record);
    this.changed();
    return this.snapshot(record);
  }

  async wait(
    afterRevision: number,
    signal?: AbortSignal,
  ): Promise<{
    revision: number;
    attacks: AttackSnapshot[];
  }> {
    if (this.revision <= afterRevision && this.hasNonTerminal()) {
      await new Promise<void>((resolveWait, reject) => {
        let settled = false;
        const finish = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          this.listeners.delete(finish);
          signal?.removeEventListener("abort", abort);
          resolveWait();
        };
        const abort = () => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          this.listeners.delete(finish);
          reject(new Error("wait for attacks aborted"));
        };
        const timer = setTimeout(finish, WAIT_TIMEOUT_MS);
        this.listeners.add(finish);
        signal?.addEventListener("abort", abort, { once: true });
      });
    }
    return { revision: this.revision, attacks: this.snapshots() };
  }

  private beginShutdown(reason: string): void {
    this.stopping = true;
    let changed = false;
    for (const record of this.records.values()) {
      if (record.status === "queued" || record.status === "running") {
        record.status = "interrupted";
        record.error = reason;
        this.controllers.get(record.id)?.abort();
        this.saveAttack(record);
        changed = true;
      }
    }
    if (changed) {
      this.changed();
    }
  }

  async shutdown(reason: string): Promise<void> {
    this.beginShutdown(reason);
    await Promise.allSettled([...this.settlements.values()]);
  }

  private requestSchedule(): void {
    if (this.stopping || this.schedulePending) {
      return;
    }
    this.schedulePending = true;
    queueMicrotask(() => {
      this.schedulePending = false;
      this.schedule();
    });
  }

  private schedule(): void {
    if (this.stopping || this.signal?.aborted) {
      return;
    }
    while (this.active < this.checkpoint.concurrency) {
      const id = this.queue.shift();
      if (!id) {
        return;
      }
      const record = this.records.get(id);
      if (record?.status !== "queued") {
        continue;
      }
      this.active++;
      record.status = "running";
      this.saveAttack(record);
      this.changed();
      const settlement = this.runAttack(record)
        .catch((error) => {
          if (record.status !== "cancelled" && record.status !== "interrupted") {
            record.status = "failed";
            record.error = error instanceof Error ? error.message : String(error);
            this.saveAttack(record);
            this.changed();
          }
        })
        .finally(() => {
          this.controllers.delete(record.id);
          this.active--;
          this.requestSchedule();
        });
      this.settlements.set(id, settlement);
      void settlement.then(
        () => this.settlements.delete(id),
        () => this.settlements.delete(id),
      );
    }
  }

  private pendingSteering(record: AttackCheckpoint): AgentMessage[] {
    const pending = record.steering.filter((instruction) => !instruction.delivered);
    return pending.map((instruction) => ({
      role: "user" as const,
      content: `Chieftain steering #${instruction.sequence}: ${instruction.content}`,
      timestamp: Date.now(),
    }));
  }

  private markPersistedSteering(record: AttackCheckpoint, messages: AgentMessage[]): void {
    const delivered = new Set<number>();
    for (const message of messages) {
      if (message.role !== "user" || typeof message.content !== "string") continue;
      const sequence = /^Chieftain steering #(\d+):/.exec(message.content)?.[1];
      if (sequence !== undefined) delivered.add(Number(sequence));
    }
    for (const instruction of record.steering) {
      if (delivered.has(instruction.sequence)) instruction.delivered = true;
    }
  }

  private async runAttack(record: AttackCheckpoint): Promise<void> {
    if (!existsSync(record.worktree)) {
      git(
        record.repo,
        barbarianWorktreeAddArgs(record.artifactRoot, record.worktree, record.target),
      );
    }
    const controller = new AbortController();
    this.controllers.set(record.id, controller);
    const signal = this.signal
      ? AbortSignal.any([this.signal, controller.signal])
      : controller.signal;
    const pi = await import("@earendil-works/pi-coding-agent");
    const context: AgentContext = {
      systemPrompt: ATTACK_SYSTEM_PROMPT,
      messages: [],
      tools: [
        pi.createReadTool(record.worktree),
        pi.createBashTool(record.worktree, {
          spawnHook: barbarianBashSpawnHook(record.artifactRoot),
        }),
        pi.createGrepTool(record.worktree),
        pi.createFindTool(record.worktree),
        pi.createLsTool(record.worktree),
        pi.createWriteTool(record.worktree),
      ],
    };

    let promptMessages: AgentMessage[];
    if (record.messages.length === 0) {
      promptMessages = [
        {
          role: "user",
          content: `ATTACK_ID: ${record.id}\nSOURCE: ${record.source}\nTARGET: ${record.target}\nARTIFACT_ROOT: ${record.artifactRoot}\nWORKTREE: ${record.worktree}\n\nHypothesis: ${record.hypothesis}\nObjective: ${record.objective}\n\nProve or refute this hypothesis. Use the isolated worktree for every write and reproduction.`,
          timestamp: Date.now(),
        },
      ];
    } else {
      promptMessages = [{ role: "user", content: ATTACK_RESUME_NUDGE, timestamp: Date.now() }];
    }

    const config: AgentLoopConfig = {
      model: resolveModel(record.modelSpec),
      ...(record.reasoning !== "off" && { reasoning: record.reasoning }),
      convertToLlm,
      getSteeringMessages: async () => this.pendingSteering(record),
    };
    const outcome = await runAgentWithRecovery({
      context,
      config,
      transcript: record.messages,
      promptMessages,
      signal,
      onEvent: (event) => {
        this.onEvent?.(record.id, event);
        if (event.type === "turn_end" && event.message.role === "assistant") {
          record.latestProgress = progressText(event.message);
          this.saveAttack(record);
          this.changed();
        }
      },
      onMessages: (messages) => {
        record.messages = messages;
        this.markPersistedSteering(record, messages);
        this.saveAttack(record);
      },
    });
    if (signal.aborted || record.status === "cancelled" || record.status === "interrupted") {
      throw new Error(`attack stopped before completion: ${record.id}`);
    }
    record.messages = outcome.messages;
    record.result = finalText(outcome.final);
    record.latestProgress = record.result;
    record.status = "completed";
    record.error = undefined;
    writeFileSync(join(record.artifactRoot, "result.md"), `${record.result}\n`);
    this.saveAttack(record);
    this.changed();
  }
}

const LAUNCH_PARAMETERS = Type.Object({
  hypothesis: Type.String({ description: "Concrete defect hypothesis to test." }),
  objective: Type.String({
    description: "Bounded evidence or reproduction the attack should obtain.",
  }),
});
const WAIT_PARAMETERS = Type.Object({
  afterRevision: Type.Integer({
    minimum: 0,
    description: "Most recent horde revision already observed.",
  }),
});
const STEER_PARAMETERS = Type.Object({
  attackId: Type.String(),
  instructions: Type.String(),
});
const CANCEL_PARAMETERS = Type.Object({ attackId: Type.String() });
const LIST_PARAMETERS = Type.Object({});

function textToolResult<T>(value: T): AgentToolResult<T> {
  return {
    content: [{ type: "text", text: JSON.stringify(value, null, 2) }],
    details: value,
  };
}

function hordeTools(coordinator: HordeCoordinator): AgentTool[] {
  const launch: AgentTool<typeof LAUNCH_PARAMETERS> = {
    name: "launch_attack",
    label: "Launch horde attack",
    description:
      "Launch an independent attack agent for one hypothesis. Call this tool multiple times " +
      "in one response to run attacks concurrently. Returns immediately with an attack id.",
    parameters: LAUNCH_PARAMETERS,
    execute: async (_id, params) => {
      const attack = coordinator.launch(params.hypothesis, params.objective);
      return textToolResult({ revision: coordinator.revision, attack });
    },
  };
  const wait: AgentTool<typeof WAIT_PARAMETERS> = {
    name: "wait_for_attacks",
    label: "Wait for horde",
    description:
      "Wait until the horde advances beyond a previously observed revision, then return all attack states.",
    parameters: WAIT_PARAMETERS,
    execute: async (_id, params, signal) =>
      textToolResult(await coordinator.wait(params.afterRevision, signal)),
  };
  const steer: AgentTool<typeof STEER_PARAMETERS> = {
    name: "steer_attack",
    label: "Steer horde attack",
    description:
      "Queue steering for an attack. Live attacks receive it between turns; completed or failed attacks reopen from their checkpoint.",
    parameters: STEER_PARAMETERS,
    execute: async (_id, params) => {
      const attack = coordinator.steer(params.attackId, params.instructions);
      return textToolResult({ revision: coordinator.revision, attack });
    },
  };
  const cancel: AgentTool<typeof CANCEL_PARAMETERS> = {
    name: "cancel_attack",
    label: "Cancel horde attack",
    description: "Cancel a queued or running attack that is no longer useful.",
    parameters: CANCEL_PARAMETERS,
    execute: async (_id, params) => {
      const attack = coordinator.cancel(params.attackId);
      return textToolResult({ revision: coordinator.revision, attack });
    },
  };
  const list: AgentTool<typeof LIST_PARAMETERS> = {
    name: "list_attacks",
    label: "List horde attacks",
    description: "Return the current revision and all attack states without waiting.",
    parameters: LIST_PARAMETERS,
    execute: async () =>
      textToolResult({ revision: coordinator.revision, attacks: coordinator.snapshots() }),
  };
  return [launch, wait, steer, cancel, list];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isReasoningLevel(value: unknown): value is ReasoningLevel {
  return typeof value === "string" && (REASONING_LEVELS as readonly string[]).includes(value);
}

function optionalString(value: unknown): boolean {
  return value === undefined || typeof value === "string";
}

function strictDescendant(parent: string, child: string): boolean {
  const rel = relative(parent, child);
  return rel !== "" && rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel);
}

function assertCheckpointPathComponents(root: string, path: string, label: string): void {
  const canonicalRoot = canonicalBarbarianArtifactRoot(root);
  if (!strictDescendant(canonicalRoot, resolve(path))) {
    throw new Error(`${label} must remain beneath ${canonicalRoot}`);
  }
  let cursor = canonicalRoot;
  for (const component of relative(canonicalRoot, resolve(path)).split(sep)) {
    cursor = join(cursor, component);
    if (!existsSync(cursor)) break;
    if (lstatSync(cursor).isSymbolicLink()) {
      throw new Error(`${label} contains a symbolic-link path component: ${cursor}`);
    }
  }
  if (existsSync(path) && !strictDescendant(canonicalRoot, canonicalBarbarianArtifactRoot(path))) {
    throw new Error(`${label} resolves outside ${canonicalRoot}`);
  }
}

function exactCheckpointPath(
  value: unknown,
  expected: string,
  label: string,
  containmentRoot?: string,
): string {
  if (typeof value !== "string" || resolve(value) !== resolve(expected)) {
    throw new Error(`${label} must be ${expected}`);
  }
  if (containmentRoot) assertCheckpointPathComponents(containmentRoot, expected, label);
  if (
    existsSync(value) &&
    canonicalBarbarianArtifactRoot(value) !== canonicalBarbarianArtifactRoot(expected)
  ) {
    throw new Error(`${label} resolves outside ${expected}`);
  }
  return resolve(expected);
}

function loadHordeCheckpoint(artifactRoot: string): HordeCheckpoint {
  const path = join(artifactRoot, REVIEW_CHECKPOINT);
  const value = readJson<unknown>(path);
  if (!isRecord(value) || value.version !== 2 || value.mode !== "horde") {
    throw new Error(`${path} is not a horde checkpoint`);
  }
  const checkpoint = value as Partial<HordeCheckpoint>;
  const strings = [
    checkpoint.repo,
    checkpoint.source,
    checkpoint.target,
    checkpoint.chieftainModelSpec,
    checkpoint.hordeModelSpec,
  ];
  if (
    strings.some((entry) => typeof entry !== "string" || entry.length === 0) ||
    !optionalString(checkpoint.syntheticTarget) ||
    !optionalString(checkpoint.instructions) ||
    !isReasoningLevel(checkpoint.chieftainReasoning) ||
    !isReasoningLevel(checkpoint.hordeReasoning) ||
    !Number.isSafeInteger(checkpoint.concurrency) ||
    (checkpoint.concurrency ?? 0) <= 0 ||
    !Array.isArray(checkpoint.chieftainMessages) ||
    !Array.isArray(checkpoint.attackIds) ||
    !checkpoint.attackIds.every((id) => typeof id === "string" && /^attack-\d{3,}$/.test(id)) ||
    new Set(checkpoint.attackIds).size !== checkpoint.attackIds.length ||
    !Number.isSafeInteger(checkpoint.nextAttackNumber) ||
    (checkpoint.nextAttackNumber ?? 0) <= 0 ||
    !Number.isSafeInteger(checkpoint.revision) ||
    (checkpoint.revision ?? -1) < 0
  ) {
    throw new Error(`${path} is not a complete horde checkpoint`);
  }
  const canonicalRoot = canonicalBarbarianArtifactRoot(artifactRoot);
  checkpoint.artifactRoot = exactCheckpointPath(
    checkpoint.artifactRoot,
    canonicalRoot,
    "checkpoint artifact root",
  );
  const canonicalRepo = canonicalBarbarianArtifactRoot(checkpoint.repo!);
  if (canonicalBarbarianArtifactRoot(repoRoot(canonicalRepo)) !== canonicalRepo) {
    throw new Error(`checkpoint repo is not a canonical Git work tree root: ${checkpoint.repo}`);
  }
  checkpoint.repo = canonicalRepo;
  checkpoint.chieftainWorktree = exactCheckpointPath(
    checkpoint.chieftainWorktree,
    join(canonicalRoot, "chieftain-worktree"),
    "checkpoint chieftain worktree",
    canonicalRoot,
  );
  assertCheckpointPathComponents(
    canonicalRoot,
    join(canonicalRoot, "attacks"),
    "checkpoint attacks directory",
  );
  const source = resolveCommit(canonicalRepo, checkpoint.source!);
  const target = resolveCommit(canonicalRepo, checkpoint.target!);
  if (source !== checkpoint.source || target !== checkpoint.target) {
    throw new Error(`${path} must store resolved source and target commit IDs`);
  }
  const highestAttack = checkpoint.attackIds.reduce(
    (highest, id) => Math.max(highest, Number(id.slice("attack-".length))),
    0,
  );
  if (checkpoint.nextAttackNumber! <= highestAttack) {
    throw new Error(`${path} next attack number would overwrite an existing attack`);
  }
  checkpoint.chieftainMessages = repairDanglingToolCalls(checkpoint.chieftainMessages);
  return checkpoint as HordeCheckpoint;
}

function loadAttacks(checkpoint: HordeCheckpoint): Map<string, AttackCheckpoint> {
  const records = new Map<string, AttackCheckpoint>();
  for (const id of checkpoint.attackIds) {
    const path = join(checkpoint.artifactRoot, "attacks", id, ATTACK_CHECKPOINT);
    const value = readJson<unknown>(path);
    if (!isRecord(value)) {
      throw new Error(`${path} is not a valid attack checkpoint`);
    }
    const record = value as Partial<AttackCheckpoint>;
    const statuses: readonly AttackStatus[] = [
      "queued",
      "running",
      "completed",
      "failed",
      "cancelled",
      "interrupted",
    ];
    if (
      record.version !== 1 ||
      record.kind !== "horde-attack" ||
      record.id !== id ||
      typeof record.hypothesis !== "string" ||
      typeof record.objective !== "string" ||
      record.repo !== checkpoint.repo ||
      record.source !== checkpoint.source ||
      record.target !== checkpoint.target ||
      typeof record.modelSpec !== "string" ||
      !isReasoningLevel(record.reasoning) ||
      !statuses.includes(record.status as AttackStatus) ||
      !Array.isArray(record.messages) ||
      !Array.isArray(record.steering) ||
      !record.steering.every(
        (instruction) =>
          isRecord(instruction) &&
          Number.isSafeInteger(instruction.sequence) &&
          (instruction.sequence as number) > 0 &&
          typeof instruction.content === "string" &&
          typeof instruction.delivered === "boolean",
      ) ||
      !Number.isSafeInteger(record.nextSteeringSequence) ||
      (record.nextSteeringSequence ?? 0) <= 0 ||
      !optionalString(record.latestProgress) ||
      !optionalString(record.result) ||
      !optionalString(record.error)
    ) {
      throw new Error(`${path} is not a complete attack checkpoint`);
    }
    record.artifactRoot = exactCheckpointPath(
      record.artifactRoot,
      join(checkpoint.artifactRoot, "attacks", id),
      `attack ${id} artifact root`,
      checkpoint.artifactRoot,
    );
    record.worktree = exactCheckpointPath(
      record.worktree,
      join(record.artifactRoot, "worktree"),
      `attack ${id} worktree`,
      checkpoint.artifactRoot,
    );
    record.messages = repairDanglingToolCalls(record.messages);
    records.set(id, record as AttackCheckpoint);
  }
  return records;
}

function prepareFresh(
  options: BarbarianOptions,
  concurrency: number,
): { checkpoint: HordeCheckpoint; release: () => void } {
  const cwd = process.cwd();
  const repoArg = options.repo?.trim() || cwd;
  const repo = repoRoot(isAbsolute(repoArg) ? repoArg : resolve(cwd, repoArg));
  const { chieftainModelSpec, chieftainReasoning, hordeModelSpec, hordeReasoning } =
    hordeIntelligence(options);
  resolveModel(chieftainModelSpec);
  resolveModel(hordeModelSpec);
  const { artifactRoot, release } = createLockedBarbarianArtifactRoot(
    "horde",
    repo,
    options.barbarianDir,
  );
  try {
    mkdirSync(join(artifactRoot, "attacks"), { recursive: true });
    const source = resolveCommit(repo, resolveSource(repo, options.source?.trim() || undefined));
    const requestedTarget = options.target?.trim();
    const syntheticTarget = requestedTarget
      ? undefined
      : hasChanges(repo)
        ? createSyntheticTarget(repo, artifactRoot)
        : undefined;
    const target = resolveCommit(repo, requestedTarget || syntheticTarget || "HEAD");
    const instructions = options.instructions?.trim();
    return {
      checkpoint: {
        version: 2,
        mode: "horde",
        repo,
        source,
        target,
        ...(syntheticTarget !== undefined && { syntheticTarget }),
        artifactRoot,
        chieftainWorktree: join(artifactRoot, "chieftain-worktree"),
        chieftainModelSpec,
        chieftainReasoning,
        hordeModelSpec,
        hordeReasoning,
        concurrency,
        ...(instructions && { instructions }),
        chieftainMessages: [],
        attackIds: [],
        nextAttackNumber: 1,
        revision: 0,
      },
      release,
    };
  } catch (error) {
    release();
    throw error;
  }
}

async function runLockedHordeReview(
  options: BarbarianOptions,
  concurrency: number,
  checkpoint: HordeCheckpoint,
): Promise<BarbarianResult> {
  // Validate every persisted path before creating or loading a worktree.
  const records = loadAttacks(checkpoint);
  checkpoint.concurrency = concurrency;
  if (options.model?.trim()) {
    checkpoint.chieftainModelSpec = options.model.trim();
  }
  if (options.reasoning) {
    checkpoint.chieftainReasoning = options.reasoning;
  }
  if (options.hordeModel?.trim()) {
    checkpoint.hordeModelSpec = options.hordeModel.trim();
  }
  if (options.hordeReasoning) {
    checkpoint.hordeReasoning = options.hordeReasoning;
  }
  resolveModel(checkpoint.chieftainModelSpec);
  resolveModel(checkpoint.hordeModelSpec);
  if (!existsSync(checkpoint.chieftainWorktree)) {
    git(
      checkpoint.repo,
      barbarianWorktreeAddArgs(
        checkpoint.artifactRoot,
        checkpoint.chieftainWorktree,
        checkpoint.target,
      ),
    );
  }
  atomicWriteJson(join(checkpoint.artifactRoot, REVIEW_CHECKPOINT), checkpoint);
  options.onArtifactRoot?.(checkpoint.artifactRoot);

  const coordinator = new HordeCoordinator({
    checkpoint,
    records,
    signal: options.signal,
    onEvent: options.onHordeEvent,
  });
  const pi = await import("@earendil-works/pi-coding-agent");
  const context: AgentContext = {
    systemPrompt: CHIEFTAIN_SYSTEM_PROMPT,
    messages: [],
    tools: [
      pi.createReadTool(checkpoint.chieftainWorktree),
      pi.createBashTool(checkpoint.chieftainWorktree, {
        spawnHook: barbarianBashSpawnHook(checkpoint.artifactRoot),
      }),
      pi.createGrepTool(checkpoint.chieftainWorktree),
      pi.createFindTool(checkpoint.chieftainWorktree),
      pi.createLsTool(checkpoint.chieftainWorktree),
      ...hordeTools(coordinator),
    ],
  };
  const freshPrompt: AgentMessage = {
    role: "user",
    content: `SOURCE: ${checkpoint.source}\nTARGET: ${checkpoint.target}\nARTIFACT_ROOT: ${checkpoint.artifactRoot}\nHORDE_CONCURRENCY: ${checkpoint.concurrency}\nHORDE_MODEL: ${checkpoint.hordeModelSpec}\n\nReview SOURCE..TARGET. Generate concrete hypotheses, launch independent attacks in parallel, steer them when useful, then synthesize the complete findings report.${checkpoint.instructions ? `\n\nExtra instructions:\n${checkpoint.instructions}` : ""}`,
    timestamp: Date.now(),
  };
  const promptMessages: AgentMessage[] =
    options.resume && checkpoint.chieftainMessages.length > 0
      ? [{ role: "user", content: CHIEFTAIN_RESUME_NUDGE, timestamp: Date.now() }]
      : [freshPrompt];
  let observedRevision = checkpoint.revision;
  let followUpAborted = false;
  const config: AgentLoopConfig = {
    model: resolveModel(checkpoint.chieftainModelSpec),
    ...(checkpoint.chieftainReasoning !== "off" && {
      reasoning: checkpoint.chieftainReasoning,
    }),
    convertToLlm,
    getFollowUpMessages: async () => {
      if (!coordinator.hasNonTerminal() && coordinator.revision <= observedRevision) {
        return [];
      }
      let update: { revision: number; attacks: AttackSnapshot[] };
      try {
        update = await coordinator.wait(observedRevision, options.signal);
      } catch {
        followUpAborted = true;
        return [];
      }
      observedRevision = update.revision;
      return [
        {
          role: "user",
          content: `Horde update (revision ${update.revision}):\n${JSON.stringify(update.attacks, null, 2)}\n\nInspect the evidence. Wait, steer, cancel, or launch follow-up attacks as appropriate. Do not finish while attacks remain queued or running.`,
          timestamp: Date.now(),
        },
      ];
    },
  };
  try {
    const outcome = await runAgentWithRecovery({
      context,
      config,
      transcript: checkpoint.chieftainMessages,
      promptMessages,
      signal: options.signal,
      onEvent: options.onEvent,
      onMessages: (messages) => {
        checkpoint.chieftainMessages = messages;
        atomicWriteJson(join(checkpoint.artifactRoot, REVIEW_CHECKPOINT), checkpoint);
      },
    });
    if (followUpAborted || options.signal?.aborted) {
      throw new Error("horde review aborted");
    }
    if (coordinator.snapshots().length === 0) {
      throw new Error("horde review completed without launching an attack");
    }
    if (coordinator.hasNonTerminal()) {
      throw new Error("horde review completed while attacks were still queued or running");
    }
    if (!coordinator.snapshots().some((attack) => attack.status === "completed")) {
      throw new Error("horde review completed without a successful attack");
    }
    await coordinator.shutdown("review completed");
    checkpoint.chieftainMessages = outcome.messages;
    atomicWriteJson(join(checkpoint.artifactRoot, REVIEW_CHECKPOINT), checkpoint);

    const usage = totalUsage([...checkpoint.chieftainMessages, ...coordinator.messages()]);
    return {
      repo: checkpoint.repo,
      source: checkpoint.source,
      target: checkpoint.target,
      artifactRoot: checkpoint.artifactRoot,
      findings: finalText(outcome.final),
      ...(checkpoint.syntheticTarget !== undefined && {
        syntheticTarget: checkpoint.syntheticTarget,
      }),
      modelSpec: checkpoint.chieftainModelSpec,
      reasoning: checkpoint.chieftainReasoning,
      concurrency: checkpoint.concurrency,
      hordeModelSpec: checkpoint.hordeModelSpec,
      hordeReasoning: checkpoint.hordeReasoning,
      usage,
    };
  } catch (error) {
    await coordinator.shutdown("chieftain stopped before review completion");
    throw error;
  }
}

export async function runBarbarianHordeReview(
  options: BarbarianOptions,
  concurrency: number,
  lockedResumeRoot?: string,
): Promise<BarbarianResult> {
  let checkpoint: HordeCheckpoint;
  let release: (() => void) | undefined;
  if (options.resume) {
    const artifactRoot = lockedResumeRoot ?? canonicalBarbarianArtifactRoot(options.resume);
    if (!lockedResumeRoot) release = acquireBarbarianRunLock(artifactRoot);
    try {
      checkpoint = loadHordeCheckpoint(artifactRoot);
      if (canonicalBarbarianArtifactRoot(checkpoint.artifactRoot) !== artifactRoot) {
        throw new Error(`checkpoint artifact root does not match --resume path: ${artifactRoot}`);
      }
    } catch (error) {
      release?.();
      throw error;
    }
  } else {
    ({ checkpoint, release } = prepareFresh(options, concurrency));
  }
  try {
    return await runLockedHordeReview(options, concurrency, checkpoint);
  } finally {
    release?.();
  }
}

export function hordeCheckpointUsage(artifactRoot: string): Usage {
  const checkpoint = loadHordeCheckpoint(artifactRoot);
  const records = loadAttacks(checkpoint);
  return totalUsage([
    ...checkpoint.chieftainMessages,
    ...[...records.values()].flatMap((record) => record.messages),
  ]);
}
