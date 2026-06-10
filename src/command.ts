/**
 * Minimal recursive CLI command framework.
 *
 * Commands form a tree; each node has its own help text. Descent peels
 * matched subcommand names: a command's `run` receives exactly the arguments
 * that follow its own name on the command line. Leaf commands parse their
 * own arguments however they like; the framework does no flag parsing.
 *
 * `-h`/`--help` anywhere on the command line prints help for the deepest
 * resolved command (the "current leaf") and exits 0.
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
   * Command handler. Receives the arguments that follow this command's name
   * (ancestor command names peeled off) and returns the process exit code.
   * Commands without a handler are non-leaf: invoking them prints help.
   */
  run?: (argv: string[]) => Promise<number> | number;
}

export function helpText(command: Command, path: string[]): string {
  const fullName = path.join(" ");
  const lines: string[] = [];
  lines.push(`${fullName} - ${command.summary}`);
  if (command.description) {
    lines.push("", command.description);
  }
  const usage =
    command.usage ??
    [
      fullName,
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

function isHelpToken(token: string): boolean {
  return token === "-h" || token === "--help";
}

/**
 * Resolve and execute a command line against a command tree.
 * Returns the process exit code.
 */
export async function execute(
  command: Command,
  argv: string[],
  path: string[] = [command.name],
): Promise<number> {
  const helpRequested = argv.some(isHelpToken);
  const args = argv.filter((token) => !isHelpToken(token));

  // Keep descending while the next token names a subcommand.
  const child = args.length > 0 ? command.subcommands?.find((c) => c.name === args[0]) : undefined;
  if (child) {
    const rest = [...args.slice(1), ...(helpRequested ? ["--help"] : [])];
    return execute(child, rest, [...path, child.name]);
  }

  if (helpRequested) {
    process.stdout.write(helpText(command, path) + "\n");
    return 0;
  }
  if (command.run) {
    return command.run(args);
  }
  if (args.length > 0) {
    process.stderr.write(`Unknown command: ${path.join(" ")} ${args[0]}\n\n`);
    process.stderr.write(helpText(command, path) + "\n");
    return 1;
  }
  process.stdout.write(helpText(command, path) + "\n");
  return 0;
}
