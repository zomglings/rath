/**
 * `rath barbarian`: run the Barbarian Reviewer against a git repository.
 *
 * A thin CLI over runBarbarianReview (src/agents/barbarian.ts): flags map
 * one-to-one onto BarbarianOptions. Non-interactive by design — progress
 * (tool calls, turn boundaries) streams to stderr, the findings report goes
 * to stdout, and the exit code is 0 only when the review completed.
 */

import { runBarbarianReview } from "../agents/barbarian.js";
import { ensureCatalogue } from "../catalogue.js";
import { type Command, fullName, helpRequested, helpText } from "../command.js";
import { registerOpenAINative, registerOpenRouterNative } from "../index.js";
import { REASONING_LEVELS, type ReasoningLevel } from "../models.js";

const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function dim(text: string): string {
  return process.stderr.isTTY ? `${DIM}${text}${RESET}` : text;
}

export const barbarianCommand: Command = {
  name: "barbarian",
  summary: "Run the Barbarian Reviewer on a git repo",
  description:
    "Reviews the changes from a source commit-ish to a target commit-ish and\n" +
    "writes a findings report. Non-interactive: the barbarian never asks\n" +
    "questions; it reads the repo, stages reproductions in disposable\n" +
    "worktrees under a temp artifact root, and reports. The report is written\n" +
    "to the findings file and printed to stdout; progress goes to stderr.\n" +
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
      long: "output",
      short: "o",
      takesValue: true,
      description:
        "Findings file path, relative to the repo root (default: findings.md in the artifact root)",
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
  ],
  async run(prefix, argv) {
    if (helpRequested(argv)) {
      process.stdout.write(`${helpText(this, prefix)}\n`);
      return 0;
    }
    let repo: string | undefined;
    let source: string | undefined;
    let target: string | undefined;
    let output: string | undefined;
    let instructions: string | undefined;
    let model: string | undefined;
    let reasoning: ReasoningLevel | undefined;
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
        } else if (token === "-o" || token === "--output") {
          output = value();
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

    try {
      const result = await runBarbarianReview({
        repo,
        source,
        target,
        output,
        instructions,
        model,
        reasoning,
        onEvent: (event) => {
          if (event.type === "agent_start") {
            process.stderr.write(dim("[barbarian] reviewing…\n"));
          } else if (event.type === "tool_execution_start") {
            const args = JSON.stringify(event.args);
            process.stderr.write(
              dim(
                `[barbarian] ${event.toolName}: ${
                  args.length > 120 ? `${args.slice(0, 120)}…` : args
                }\n`,
              ),
            );
          } else if (event.type === "message_end" && event.message.role === "assistant") {
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
          } | model: ${result.modelSpec} | reasoning: ${result.reasoning}\n` +
            `[barbarian] findings: ${result.findingsPath}\n` +
            `[barbarian] artifacts: ${result.artifactRoot}\n`,
        ),
      );
      process.stdout.write(`${result.findings}\n`);
      return 0;
    } catch (error) {
      process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
      return 1;
    }
  },
};
