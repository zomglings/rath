/**
 * `rath barbarian`: the Barbarian Reviewer CLI. A parent command with two
 * subcommands — `run` (review a git range) and `skill` (print/install the
 * Agent Skill that teaches another agent how to drive it).
 *
 * `rath barbarian run` is a thin wrapper over runBarbarianReview
 * (src/agents/barbarian.ts). Non-interactive by design — progress streams to
 * stderr (the reasoning summary and reply tokens as they generate, plus tool
 * calls), the findings report goes to stdout, and the exit code is 0 only when
 * the review completed.
 */

import { hasCheckpoint, runBarbarianReview } from "../agents/barbarian.js";
import { barbarianSkillCommand } from "../barbarian-skill.js";
import { ensureCatalogue } from "../catalogue.js";
import { type Command, fullName, helpRequested, helpText, runSubcommands } from "../command.js";
import { registerOpenAINative, registerOpenRouterNative } from "../index.js";
import { REASONING_LEVELS, type ReasoningLevel } from "../models.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function dim(text: string): string {
  return process.stderr.isTTY ? `${DIM}${text}${RESET}` : text;
}

const barbarianRunCommand: Command = {
  name: "run",
  summary: "Run a review on a git range",
  description:
    "Reviews the changes from a source commit-ish to a target commit-ish and\n" +
    "reports its findings. Non-interactive: the barbarian never asks\n" +
    "questions; it reads the repo, stages reproductions in disposable\n" +
    "worktrees under a temp artifact root, and reports. The report prints to\n" +
    "stdout (redirect to save it); progress goes to stderr.\n" +
    "\n" +
    "Defaults: source is main (master if no main); target is the current\n" +
    "repository state including staged, unstaged, and untracked changes,\n" +
    "captured as a synthetic commit in a disposable worktree (your tree is\n" +
    "never touched). Requires the provider API key in the environment.",
  flags: [
    {
      long: "repo",
      short: "r",
      takesValue: true,
      description: "Target git repository (default: the current working directory)",
    },
    {
      long: "source",
      short: "s",
      takesValue: true,
      description: "Source commit-ish (default: main, falling back to master)",
    },
    {
      long: "target",
      short: "t",
      takesValue: true,
      description:
        "Target commit-ish (default: the current repo state, including uncommitted changes)",
    },
    {
      long: "instructions",
      short: "i",
      takesValue: true,
      description: "Extra reviewer instructions",
    },
    {
      long: "model",
      short: "m",
      takesValue: true,
      description: "Barbarian's model as <provider>/<model-id> (default: the pinned default model)",
    },
    {
      long: "reasoning",
      takesValue: true,
      description: `Barbarian's reasoning effort: ${REASONING_LEVELS.join(", ")} (default: high)`,
    },
    {
      long: "resume",
      takesValue: true,
      description:
        "Resume an interrupted review from its artifact root (the 'artifacts:' " +
        "path printed by the original run); reuses its range and transcript",
    },
  ],
  async run(prefix, argv) {
    if (helpRequested(argv)) {
      process.stdout.write(`${helpText(this, prefix)}\n`);
      return 0;
    }
    let repo: string | undefined;
    let source: string | undefined;
    let target: string | undefined;
    let instructions: string | undefined;
    let model: string | undefined;
    let reasoning: ReasoningLevel | undefined;
    let resume: string | undefined;
    for (let i = 0; i < argv.length; i++) {
      const token = argv[i]!;
      const value = (): string => {
        const v = argv[++i];
        if (v === undefined) {
          throw new Error(`Option ${token} requires a value`);
        }
        return v;
      };
      try {
        if (token === "-r" || token === "--repo") {
          repo = value();
        } else if (token === "-s" || token === "--source") {
          source = value();
        } else if (token === "-t" || token === "--target") {
          target = value();
        } else if (token === "-i" || token === "--instructions") {
          instructions = value();
        } else if (token === "-m" || token === "--model") {
          model = value();
        } else if (token === "--reasoning") {
          const level = value() as ReasoningLevel;
          if (!REASONING_LEVELS.includes(level)) {
            process.stderr.write(
              `Invalid --reasoning: ${level} (use ${REASONING_LEVELS.join(", ")})\n`,
            );
            return 1;
          }
          reasoning = level;
        } else if (token === "--resume") {
          resume = value();
        } else {
          process.stderr.write(
            `Unknown argument: ${token}\nRun "${fullName(this, prefix)} -h" for usage.\n`,
          );
          return 1;
        }
      } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
        return 1;
      }
    }

    registerOpenAINative();
    registerOpenRouterNative();
    // Best-effort live catalogue so openrouter-native model specs resolve.
    await ensureCatalogue();

    // Stream the barbarian's reasoning summary and reply tokens to stderr as
    // they arrive, alongside the tool-call progress. `streaming` tracks whether
    // we are mid-line (no trailing newline yet) so a following progress line
    // starts cleanly. The final report still goes to stdout in full.
    let streaming = false;
    const endStreamLine = () => {
      if (streaming) {
        process.stderr.write("\n");
        streaming = false;
      }
    };
    // Captured the moment the artifact root is known (before the first turn), so
    // a failed run still tells the user where to --resume from.
    let artifactRoot: string | undefined;
    try {
      const result = await runBarbarianReview({
        repo,
        source,
        target,
        instructions,
        model,
        reasoning,
        resume,
        onArtifactRoot: (root) => {
          artifactRoot = root;
          process.stderr.write(dim(`[barbarian] artifacts: ${root}\n`));
        },
        onEvent: (event) => {
          if (event.type === "agent_start") {
            process.stderr.write(dim("[barbarian] reviewing…\n"));
          } else if (event.type === "message_update") {
            // assistantMessageEvent carries the per-token deltas: thinking_delta
            // is the reasoning summary, text_delta is the decoder output.
            const e = event.assistantMessageEvent;
            switch (e.type) {
              case "thinking_start":
                endStreamLine();
                process.stderr.write(dim("[barbarian] thinking: "));
                streaming = true;
                break;
              case "thinking_delta":
                process.stderr.write(dim(e.delta));
                break;
              case "text_start":
                endStreamLine();
                process.stderr.write(dim("[barbarian] "));
                streaming = true;
                break;
              case "text_delta":
                // Decoder tokens, undimmed so the content stands out.
                process.stderr.write(e.delta);
                break;
              case "thinking_end":
              case "text_end":
                endStreamLine();
                break;
            }
          } else if (event.type === "tool_execution_start") {
            endStreamLine();
            const args = JSON.stringify(event.args);
            process.stderr.write(
              dim(
                `[barbarian] ${event.toolName}: ${
                  args.length > 120 ? `${args.slice(0, 120)}…` : args
                }\n`,
              ),
            );
          } else if (event.type === "message_end" && event.message.role === "assistant") {
            endStreamLine();
            const m = event.message;
            if (m.stopReason === "error" || m.stopReason === "aborted") {
              process.stderr.write(dim(`[barbarian] ${m.stopReason}: ${m.errorMessage ?? ""}\n`));
            }
          }
        },
      });
      process.stderr.write(
        dim(
          `[barbarian] ${result.source} -> ${
            result.syntheticTarget ? `${result.target} (synthetic)` : result.target
          } | model: ${result.modelSpec} | reasoning: ${result.reasoning}\n`,
        ),
      );
      process.stdout.write(`${result.findings}\n`);
      return 0;
    } catch (error) {
      endStreamLine();
      process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
      // The artifact root was printed when the run started; repeat the resume
      // hint here so a failed run shows exactly how to pick up where it stopped.
      // Only suggest --resume when a checkpoint actually exists: a pre-flight
      // failure (bad model spec, synthetic-target error) leaves an artifact root
      // with no checkpoint, and resuming it would only fail.
      if (artifactRoot && hasCheckpoint(artifactRoot)) {
        process.stderr.write(
          dim(`[barbarian] resume with: rath barbarian run --resume ${artifactRoot}\n`),
        );
      }
      return 1;
    }
  },
};

export const barbarianCommand: Command = {
  name: "barbarian",
  summary: "The Barbarian Reviewer (run reviews, or install its skill)",
  description:
    "The Barbarian Reviewer is its own agent: a relentless, non-interactive\n" +
    "code reviewer that adversarially attacks a git diff and reports defects.\n" +
    "It is a program, not a tool — a human or another agent invokes it.\n" +
    "\n" +
    "  rath barbarian run     review a git range (see `run -h`)\n" +
    "  rath barbarian skill   print or install the Agent Skill that teaches\n" +
    "                         another agent how to drive `rath barbarian run`",
  subcommands: [barbarianRunCommand, barbarianSkillCommand],
  run: runSubcommands,
};
