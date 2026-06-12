/**
 * The `barbarian_review` tool: lets a rath session spawn the Barbarian
 * Reviewer (src/agents/barbarian.ts) against a git repository. A thin
 * AgentTool wrapper over runBarbarianReview — same defaults as the CLI
 * (`rath barbarian`): source falls back main -> master, target defaults to
 * the current repository state captured as a synthetic commit, and the
 * caller may choose the barbarian's model and reasoning level.
 *
 * The review runs in-process and unattended; in slow mode the spawn itself
 * is gated behind the per-call confirmation like any other tool.
 */
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { type BarbarianResult, runBarbarianReview } from "../agents/barbarian.js";
import { REASONING_LEVELS, type ReasoningLevel } from "../models.js";

const PARAMETERS = Type.Object({
  repo: Type.Optional(
    Type.String({
      description: "Target git repository. Defaults to the current working directory.",
    }),
  ),
  source: Type.Optional(
    Type.String({
      description: "Source commit-ish. Defaults to main if it exists, otherwise master.",
    }),
  ),
  target: Type.Optional(
    Type.String({
      description:
        "Target commit-ish. Defaults to the current repository state, including staged, " +
        "unstaged, and untracked changes (captured as a synthetic commit; the working tree " +
        "is not touched).",
    }),
  ),
  output: Type.Optional(
    Type.String({
      description:
        "Findings file path. Relative paths resolve against the repo root. Defaults to " +
        "findings.md under the review's artifact root.",
    }),
  ),
  instructions: Type.Optional(
    Type.String({ description: "Extra reviewer instructions appended to the prompt." }),
  ),
  model: Type.Optional(
    Type.String({
      description:
        "The barbarian's model, as <provider>/<model-id>. Defaults to the pinned default model.",
    }),
  ),
  reasoning: Type.Optional(
    Type.Union(
      REASONING_LEVELS.map((level) => Type.Literal(level)),
      { description: "The barbarian's reasoning effort. Defaults to high." },
    ),
  ),
});

export type BarbarianReviewDetails = Omit<BarbarianResult, "findings">;

export function createBarbarianReviewTool(): AgentTool<typeof PARAMETERS, BarbarianReviewDetails> {
  return {
    name: "barbarian_review",
    label: "Barbarian review",
    description:
      "Run the non-interactive Barbarian Reviewer against a git repo and get its findings " +
      "report. By default reviews main (or master) against the current repository state, " +
      "including staged, unstaged, and untracked changes. The report is returned and written " +
      "to the findings file; reproduction artifacts land under a temp artifact root. " +
      "Optionally choose the barbarian's model and reasoning level.",
    parameters: PARAMETERS,
    execute: async (_toolCallId, params): Promise<AgentToolResult<BarbarianReviewDetails>> => {
      const result = await runBarbarianReview({
        repo: params.repo,
        source: params.source,
        target: params.target,
        output: params.output,
        instructions: params.instructions,
        model: params.model,
        reasoning: params.reasoning as ReasoningLevel | undefined,
      });
      const { findings, ...details } = result;
      return {
        content: [
          {
            type: "text",
            text:
              `Barbarian review complete (${details.modelSpec}, reasoning ${details.reasoning}).\n` +
              `Reviewed: ${details.source} -> ${details.target}` +
              `${details.syntheticTarget ? " (synthetic target from working tree)" : ""}\n` +
              `Findings file: ${details.findingsPath}\nArtifacts: ${details.artifactRoot}\n\n` +
              findings,
          },
        ],
        details,
      };
    },
  };
}
