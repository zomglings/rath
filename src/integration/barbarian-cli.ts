/** Deterministic contract test for the explicit Barbarian CLI modes. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const cli = join(dirname(dirname(fileURLToPath(import.meta.url))), "cli.js");

function run(args: string[]): { status: number | null; stdout: string; stderr: string } {
  const result = spawnSync(process.execPath, [cli, ...args], { encoding: "utf8" });
  if (result.error) throw result.error;
  return { status: result.status, stdout: result.stdout, stderr: result.stderr };
}

const parent = run(["barbarian", "-h"]);
assert.equal(parent.status, 0);
assert.match(parent.stdout, /solo\s+Review with one intelligence/);
assert.match(parent.stdout, /horde\s+Review with a parallel chieftain and horde/);
assert.match(parent.stdout, /clean\s+Remove retained Barbarian runs/);
assert.doesNotMatch(parent.stdout, /^\s*run\s/m);

const solo = run(["barbarian", "solo", "-h"]);
assert.equal(solo.status, 0);
assert.doesNotMatch(solo.stdout, /--concurrency/);
assert.doesNotMatch(solo.stdout, /--horde-model/);
assert.match(solo.stdout, /--barbarian-dir/);

const horde = run(["barbarian", "horde", "-h"]);
assert.equal(horde.status, 0);
assert.match(horde.stdout, /--concurrency <value>.*default: 4/);
assert.match(horde.stdout, /--horde-model <value>/);
assert.match(horde.stdout, /openrouter-native\/openai\/gpt-5\.6-sol/);
assert.match(horde.stdout, /openrouter-native\/openai\/gpt-5\.6-terra/);
assert.match(horde.stdout, /--horde-reasoning <value>.*default: medium/);

const clean = run(["barbarian", "clean", "-h"]);
assert.equal(clean.status, 0);
assert.match(clean.stdout, /--repo <value>/);
assert.match(clean.stdout, /--barbarian-dir <value>/);
assert.match(clean.stdout, /--dry-run/);

const zero = run(["barbarian", "horde", "--concurrency", "0"]);
assert.equal(zero.status, 1);
assert.match(zero.stderr, /use a positive integer/);

const old = run(["barbarian", "run"]);
assert.equal(old.status, 1);
assert.match(old.stderr, /Unknown command: rath barbarian run/);

process.stdout.write("Explicit solo/horde CLI modes verified; run fallback is absent.\n");
