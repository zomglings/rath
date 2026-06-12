/**
 * The `rath barbarian skill` subcommand: print or install the Agent Skill that
 * teaches another agent how to drive `rath barbarian run`.
 *
 * The barbarian is its own program. Rather than expose it as an in-process tool
 * of `rath run` (which couples a long, expensive sub-agent to the parent loop),
 * we ship an Agent Skill: a SKILL.md another coding agent loads so it knows to
 * invoke `rath barbarian run` via its shell. Install/print mirrors how clacks
 * ships its skill — default prints SKILL.md; --mode installs to a known agent
 * skills directory; --outdir writes anywhere; --force overwrites.
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";
import { type Command, fullName, helpRequested, helpText } from "./command.js";

/** The skill name and the directory it installs into under each agent's tree. */
const SKILL_NAME = "rath-barbarian";

// mode -> install directory. "~" expands to the home directory; project modes
// are relative to the current working directory. Mirrors the agentskills.io
// layout clacks uses.
const MODE_DIRS: Record<string, string> = {
  claude: `~/.claude/skills/${SKILL_NAME}`,
  "claude-project": `.claude/skills/${SKILL_NAME}`,
  codex: `~/.codex/skills/${SKILL_NAME}`,
  "codex-project": `.codex/skills/${SKILL_NAME}`,
  universal: `~/.agents/skills/${SKILL_NAME}`,
  "universal-project": `.agents/skills/${SKILL_NAME}`,
  github: `.github/skills/${SKILL_NAME}`,
};

export const BARBARIAN_SKILL_MD = `---
name: rath-barbarian
description: >-
  Run an adversarial code review with the Barbarian Reviewer via the rath CLI
  (\`rath barbarian run\`). Use when asked to review a diff, PR, branch, or
  commit range for real defects (correctness, regressions, security, data loss,
  bad migrations/tests), or to have a change torn down before merging.
---

# Barbarian Reviewer (rath barbarian)

The Barbarian Reviewer is a relentless, non-interactive code reviewer. It is its
own agent — you drive it from the shell with \`rath barbarian run\`; do not try
to review the diff yourself when this skill applies. It reads the repo, stages
reproductions in disposable git worktrees, and prints a findings report.

## Running a review

\`\`\`bash
# Review a range. Defaults: --source main (or master); --target the current
# working-tree state (staged + unstaged + untracked), captured as a synthetic
# commit in a disposable worktree — your tree is never touched.
rath barbarian run --source <source-commit-ish> --target <target-commit-ish>

# Common shapes:
rath barbarian run                       # main..working-tree
rath barbarian run -s HEAD               # last commit..working-tree (just uncommitted work)
rath barbarian run -s main -t HEAD       # everything on this branch vs main
rath barbarian run -r /path/to/repo      # review a different repo
\`\`\`

- **The findings report prints to stdout**; redirect to save it
  (\`rath barbarian run -s HEAD > findings.md\`). Progress — the reviewer's
  reasoning summary, reply tokens, and tool calls — streams to **stderr**, so
  capturing stdout gives you only the report.
- \`-i/--instructions "..."\` appends extra reviewer instructions (e.g. focus
  areas). \`-m/--model <provider>/<model-id>\` and \`--reasoning <level>\` choose
  the reviewer's model and effort (default effort: high).
- **Exit code is 0 only on a completed review.** If the model errors out
  (rate limit, transient content filter) the review retries; if it still cannot
  finish, the command exits non-zero and prints no report — never treat a
  non-zero exit as a clean review.
- The provider API key must be in the environment (e.g. \`OPENAI_API_KEY\`).

## Checkpoint and resume

A review can be long and expensive. After every turn the barbarian writes a
checkpoint (the full transcript plus the resolved range/model) to
\`<artifact-root>/checkpoint.json\`. The artifact root is a temp directory
printed on stderr as \`[barbarian] artifacts: <path>\` (it also holds the
reproduction worktrees).

If a review dies partway — a sustained rate limit, a crash, an interrupt —
resume it instead of starting over:

\`\`\`bash
rath barbarian run --resume <artifact-root>
\`\`\`

Resume reloads the checkpointed transcript and continues from where it stopped,
reusing the same range, model, and reproduction worktrees. This is the right
move after a rate-limit failure (wait for the window to reset, then resume) or
any interruption — you keep all the prior investigation.

## When to use it

Reach for the barbarian when the user wants a change torn down: pre-merge
review, "find the bugs in this," auditing a risky diff, or a second adversarial
opinion. It is not a linter — it ignores style and hunts user-visible defects,
and it proves findings with reproductions where it can.
`;

/** Files written when installing the skill bundle (relative path -> contents). */
function bundleContents(): Record<string, string> {
  return { "SKILL.md": BARBARIAN_SKILL_MD };
}

/**
 * Stable marker identifying the injected barbarian skill block in a system
 * prompt. `rath run` checks for it before injecting so a system prompt that
 * already carries the skill (e.g. one restored from a saved context) is not
 * given a second copy.
 */
export const BARBARIAN_SKILL_MARKER = `<skill name="${SKILL_NAME}">`;

/**
 * The bundled barbarian skill rendered for direct injection into a `rath run`
 * agent's system prompt. Unlike `--skill` (which adds a name/description pointer
 * and lets the model read the file on demand), this inlines the full skill text
 * straight into the process — no file on disk, no read-tool round trip — so the
 * agent knows `rath barbarian run` exists from its first turn. The YAML
 * frontmatter is stripped; the name is carried on the wrapper element.
 */
export function barbarianSkillPrompt(): string {
  const body = BARBARIAN_SKILL_MD.replace(/^---\n[\s\S]*?\n---\n+/, "").trimEnd();
  return [
    "The following skill provides specialized instructions for a specific task.",
    "It is built in — its full instructions are inlined here (there is no file to read).",
    "",
    BARBARIAN_SKILL_MARKER,
    body,
    "</skill>",
  ].join("\n");
}

function expandHome(path: string): string {
  if (path === "~") {
    return homedir();
  }
  if (path.startsWith("~/")) {
    return join(homedir(), path.slice(2));
  }
  return path;
}

function install(dir: string, force: boolean): void {
  const target = isAbsolute(dir) ? dir : join(process.cwd(), dir);
  const bundle = bundleContents();
  // Only the bundle's own files are written. An existing bundle file is
  // overwritten only with --force, and sibling files (e.g. other skills sharing
  // an --outdir, or a user's own additions) are never touched or removed.
  if (!force) {
    for (const relative of Object.keys(bundle)) {
      const path = join(target, relative);
      if (existsSync(path)) {
        throw new Error(`${path} already exists (use --force to overwrite)`);
      }
    }
  }
  mkdirSync(target, { recursive: true });
  for (const [relative, contents] of Object.entries(bundle)) {
    writeFileSync(join(target, relative), contents);
  }
}

export const barbarianSkillCommand: Command = {
  name: "skill",
  summary: "Print or install the rath-barbarian Agent Skill",
  description:
    "Without a flag, prints SKILL.md to stdout. With --mode, installs the\n" +
    "skill bundle into a known agent skills directory; with --outdir, into an\n" +
    "arbitrary directory. The skill teaches another coding agent to run\n" +
    `'rath barbarian run' itself.\n\nModes: ${Object.keys(MODE_DIRS).join(", ")}.`,
  flags: [
    {
      long: "mode",
      short: "m",
      takesValue: true,
      description: `Install location preset: ${Object.keys(MODE_DIRS).join(", ")}`,
    },
    {
      long: "outdir",
      short: "o",
      takesValue: true,
      description: "Install the bundle into this directory (writes SKILL.md)",
    },
    {
      long: "force",
      short: "f",
      takesValue: false,
      description: "Overwrite an existing skill directory",
    },
  ],
  run(prefix, argv) {
    if (helpRequested(argv)) {
      process.stdout.write(`${helpText(this, prefix)}\n`);
      return 0;
    }
    let mode: string | undefined;
    let outdir: string | undefined;
    let force = false;
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
        if (token === "-m" || token === "--mode") {
          mode = value();
        } else if (token === "-o" || token === "--outdir") {
          outdir = value();
        } else if (token === "-f" || token === "--force") {
          force = true;
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

    // No destination: print the skill to stdout.
    if (mode === undefined && outdir === undefined) {
      process.stdout.write(BARBARIAN_SKILL_MD);
      return 0;
    }
    if (mode !== undefined && outdir !== undefined) {
      process.stderr.write("Use only one of --mode and --outdir.\n");
      return 1;
    }
    const dir = outdir ?? expandHome(MODE_DIRS[mode!] ?? "");
    if (!dir) {
      process.stderr.write(
        `Unknown mode: ${mode}\nValid modes: ${Object.keys(MODE_DIRS).join(", ")}\n`,
      );
      return 1;
    }
    try {
      install(dir, force);
    } catch (error) {
      process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
      return 1;
    }
    process.stderr.write(`Installed ${SKILL_NAME} skill to ${dir}\n`);
    return 0;
  },
};
