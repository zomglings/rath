/**
 * Minimal recursive CLI command framework.
 *
 * - Commands form a tree; each node has its own help text.
 * - `-h`/`--help` anywhere on the command line prints help for the deepest
 *   resolved command (the "current leaf") and exits 0.
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

export interface ParsedFlags {
  /** Flag long name -> values (booleans collect "true"). */
  values: Map<string, string[]>;
}

export interface Command {
  name: string;
  /** One-line summary, shown in parent help. */
  summary: string;
  /** Longer description, shown in this command's own help. */
  description?: string;
  /** Usage line override; defaults to a derived one. */
  usage?: string;
  flags?: FlagSpec[];
  subcommands?: Command[];
  /** Leaf action. Intermediate commands may omit it (help is shown instead). */
  run?: (flags: ParsedFlags, positional: string[]) => Promise<number> | number;
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

/**
 * Resolve and execute a command line against a command tree.
 * Returns the process exit code.
 */
export async function execute(root: Command, argv: string[]): Promise<number> {
  // Resolve the command path: descend while leading positional tokens name
  // subcommands. Flags are collected for the resolved command.
  let command = root;
  const path = [root.name];
  let helpRequested = false;
  const rest: string[] = [];

  let i = 0;
  let descending = true;
  while (i < argv.length) {
    const token = argv[i]!;
    if (token === "-h" || token === "--help") {
      helpRequested = true;
      i++;
      continue;
    }
    if (descending && !token.startsWith("-")) {
      const child = command.subcommands?.find((c) => c.name === token);
      if (child) {
        command = child;
        path.push(child.name);
        i++;
        continue;
      }
    }
    descending = false;
    rest.push(token);
    i++;
  }

  if (helpRequested) {
    process.stdout.write(helpText(command, path) + "\n");
    return 0;
  }

  // Parse flags for the resolved command.
  const specs = command.flags ?? [];
  const flags: ParsedFlags = { values: new Map() };
  const positional: string[] = [];
  for (let j = 0; j < rest.length; j++) {
    const token = rest[j]!;
    if (token.startsWith("-") && token !== "-" && token !== "--") {
      const isLong = token.startsWith("--");
      const name = isLong ? token.slice(2) : token.slice(1);
      const spec = specs.find((s) => (isLong ? s.long === name : s.short === name));
      if (!spec) {
        process.stderr.write(`Unknown option for "${path.join(" ")}": ${token}\n\n`);
        process.stderr.write(helpText(command, path) + "\n");
        return 1;
      }
      let value = "true";
      if (spec.takesValue) {
        const next = rest[j + 1];
        if (next === undefined) {
          process.stderr.write(`Option ${token} requires a value\n`);
          return 1;
        }
        value = next;
        j++;
      }
      const existing = flags.values.get(spec.long);
      if (existing) {
        if (!spec.repeatable) {
          process.stderr.write(`Option --${spec.long} may only be given once\n`);
          return 1;
        }
        existing.push(value);
      } else {
        flags.values.set(spec.long, [value]);
      }
    } else {
      positional.push(token);
    }
  }

  if (!command.run) {
    if (positional.length > 0) {
      process.stderr.write(`Unknown command: ${path.join(" ")} ${positional[0]}\n\n`);
      process.stderr.write(helpText(command, path) + "\n");
      return 1;
    }
    process.stdout.write(helpText(command, path) + "\n");
    return 0;
  }
  return command.run(flags, positional);
}
