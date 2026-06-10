/**
 * Minimal recursive CLI command framework.
 *
 * Commands form a tree and `run` is a method: invoking the CLI is just
 * `root.run("", process.argv.slice(2))`. `run` receives the prefix of
 * ancestor command names ("" for the root) and the arguments that follow the
 * command's own name. Non-leaf commands use `runSubcommands`, which cases on
 * the first argument and delegates to the matching subcommand's `run`; leaf
 * commands parse their argv themselves — the framework does no flag parsing.
 *
 * `-h`/`--help` anywhere on the command line prints help for the deepest
 * resolved command: dispatchers pass help tokens through during descent, and
 * every `run` answers them for itself (see `helpRequested`).
 */

export interface FlagSpec {
  /** Long form without dashes, e.g. "name". */
  long: string;
  /** Single-character short form without dash, e.g. "n". */
  short?: string;
  /** Whether the flag takes a value. */
  takesValue: boolean;
  /** Whether the flag may be repeated (values accumulate). */
  repeatable?: boolean;
  description: string;
}

export interface Command {
  name: string;
  /** One-line summary, shown in parent help. */
  summary: string;
  /** Longer description, shown in this command's own help. */
  description?: string;
  /** Usage line override; defaults to a derived one. */
  usage?: string;
  /** Flag documentation for help text. Parsing is the command's own job. */
  flags?: FlagSpec[];
  subcommands?: Command[];
  /**
   * Execute this command. `prefix` is the space-joined names of ancestor
   * commands ("" for the root); `argv` is everything on the command line
   * after this command's own name. Returns the process exit code.
   */
  run(this: Command, prefix: string, argv: string[]): Promise<number> | number;
}

export function fullName(command: Command, prefix: string): string {
  return prefix ? `${prefix} ${command.name}` : command.name;
}

function isHelpToken(token: string): boolean {
  return token === "-h" || token === "--help";
}

/** Whether the command line asks for help (`-h`/`--help` anywhere). */
export function helpRequested(argv: string[]): boolean {
  return argv.some(isHelpToken);
}

export function helpText(command: Command, prefix: string): string {
  const name = fullName(command, prefix);
  const lines: string[] = [];
  lines.push(`${name} - ${command.summary}`);
  if (command.description) {
    lines.push("", command.description);
  }
  const usage =
    command.usage ??
    [
      name,
      command.subcommands?.length ? "<command>" : null,
      command.flags?.length ? "[options]" : null,
    ]
      .filter(Boolean)
      .join(" ");
  lines.push("", "Usage:", `  ${usage}`);
  if (command.subcommands?.length) {
    lines.push("", "Commands:");
    const width = Math.max(...command.subcommands.map((c) => c.name.length));
    for (const sub of command.subcommands) {
      lines.push(`  ${sub.name.padEnd(width)}  ${sub.summary}`);
    }
  }
  const flags = [...(command.flags ?? []), HELP_FLAG];
  lines.push("", "Options:");
  const rendered = flags.map((f) => ({
    label: `${f.short ? `-${f.short}, ` : "    "}--${f.long}${f.takesValue ? " <value>" : ""}`,
    description: f.description + (f.repeatable ? " (repeatable)" : ""),
  }));
  const width = Math.max(...rendered.map((r) => r.label.length));
  for (const r of rendered) {
    lines.push(`  ${r.label.padEnd(width)}  ${r.description}`);
  }
  return lines.join("\n");
}

const HELP_FLAG: FlagSpec = {
  long: "help",
  short: "h",
  takesValue: false,
  description: "Print help for the current command",
};

/**
 * Stock `run` for non-leaf commands. If the first argument (ignoring help
 * tokens) names a subcommand, delegates to its `run` with that name peeled
 * off and help tokens preserved; otherwise prints help (exit 0, or exit 1
 * with an error if an unknown command was given).
 */
export const runSubcommands: Command["run"] = function (prefix, argv) {
  const firstArg = argv.findIndex((token) => !isHelpToken(token));
  if (firstArg >= 0) {
    const child = this.subcommands?.find((c) => c.name === argv[firstArg]);
    if (child) {
      return child.run(fullName(this, prefix), [
        ...argv.slice(0, firstArg),
        ...argv.slice(firstArg + 1),
      ]);
    }
  }
  if (helpRequested(argv)) {
    process.stdout.write(helpText(this, prefix) + "\n");
    return 0;
  }
  if (firstArg >= 0) {
    process.stderr.write(`Unknown command: ${fullName(this, prefix)} ${argv[firstArg]}\n\n`);
    process.stderr.write(helpText(this, prefix) + "\n");
    return 1;
  }
  process.stdout.write(helpText(this, prefix) + "\n");
  return 0;
};
