/**
 * `rath test`: run integration tests.
 *
 * Integration tests live in src/integration/ and compile to dist/integration/.
 * Each test is a standalone script run with `node`; exit code 0 means pass,
 * non-zero means fail. Test names are the script filenames without extension.
 */
import { spawn } from "node:child_process";
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { fullName, helpRequested, helpText, type Command } from "../command.js";

interface IntegrationTest {
  name: string;
  file: string;
}

function integrationDir(): string {
  // dist/commands/test.js -> dist/integration
  return join(dirname(dirname(fileURLToPath(import.meta.url))), "integration");
}

export function discoverTests(): IntegrationTest[] {
  const dir = integrationDir();
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  return entries
    .filter((f) => f.endsWith(".js") && !f.endsWith(".d.js"))
    .sort()
    .map((f) => ({ name: f.replace(/\.js$/, ""), file: join(dir, f) }));
}

function runTest(test: IntegrationTest): Promise<number> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [test.file], { stdio: "inherit" });
    child.on("error", reject);
    child.on("close", (code, signal) => resolve(signal ? 1 : (code ?? 1)));
  });
}

export const testCommand: Command = {
  name: "test",
  summary: "Run integration tests",
  description:
    "Runs integration test scripts from the package's integration directory.\n" +
    "Each test exits 0 on success and non-zero on failure. Some tests call\n" +
    "external APIs and require credentials (e.g. OPENAI_API_KEY) in the\n" +
    "environment.",
  flags: [
    {
      long: "name",
      short: "n",
      takesValue: true,
      repeatable: true,
      description: "Run only the named test",
    },
    {
      long: "list",
      takesValue: false,
      description: "List available tests without running them",
    },
  ],
  async run(prefix, argv) {
    if (helpRequested(argv)) {
      process.stdout.write(helpText(this, prefix) + "\n");
      return 0;
    }
    const names: string[] = [];
    let list = false;
    for (let i = 0; i < argv.length; i++) {
      const token = argv[i]!;
      if (token === "-n" || token === "--name") {
        const value = argv[++i];
        if (value === undefined) {
          process.stderr.write(`Option ${token} requires a value\n`);
          return 1;
        }
        names.push(value);
      } else if (token === "--list") {
        list = true;
      } else {
        process.stderr.write(
          `Unknown argument: ${token}\nRun "${fullName(this, prefix)} -h" for usage.\n`,
        );
        return 1;
      }
    }

    const all = discoverTests();
    if (list) {
      for (const test of all) {
        process.stdout.write(test.name + "\n");
      }
      return 0;
    }
    let selected = all;
    if (names.length > 0) {
      const known = new Map(all.map((t) => [t.name, t]));
      const missing = names.filter((n) => !known.has(n));
      if (missing.length > 0) {
        process.stderr.write(`Unknown test(s): ${missing.join(", ")}\n`);
        process.stderr.write(`Available: ${all.map((t) => t.name).join(", ") || "(none)"}\n`);
        return 1;
      }
      selected = names.map((n) => known.get(n)!);
    }
    if (selected.length === 0) {
      process.stdout.write("No integration tests found.\n");
      return 0;
    }
    const results: { name: string; code: number }[] = [];
    for (const test of selected) {
      process.stdout.write(`=== ${test.name} ===\n`);
      const code = await runTest(test);
      results.push({ name: test.name, code });
      process.stdout.write(`=== ${test.name}: ${code === 0 ? "PASS" : `FAIL (exit ${code})`} ===\n`);
    }
    const failed = results.filter((r) => r.code !== 0);
    process.stdout.write(
      `\n${results.length - failed.length}/${results.length} passed` +
        (failed.length > 0 ? `; failed: ${failed.map((f) => f.name).join(", ")}` : "") +
        "\n",
    );
    return failed.length > 0 ? 1 : 0;
  },
};
