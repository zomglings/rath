/**
 * Integration test: the SQLite-backed config store. Calls no API and needs no
 * key — it points $RATH_CONFIG_DIR at a temp directory and exercises the store
 * directly. Checks:
 *  1. configDir() honors $RATH_CONFIG_DIR, and the default path is
 *     platform-appropriate and ends in "rath".
 *  2. A fresh store reads as empty; setDefaultModel writes, updates, and
 *     clears the pinned default model (round-trip through a real DB file).
 *  3. The schema migrates automatically on open: user_version reaches the
 *     migration count and the preferences table exists, and re-opening is
 *     idempotent (no error, no version drift).
 *  4. A store whose user_version is ahead of this build is left untouched
 *     (forward-compatible: never downgrades).
 *
 * Exits 0 on success, 1 on failure. Skipped when node:sqlite is unavailable.
 */
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { clearDefaultModel, configDir, loadPreferences, setDefaultModel } from "../index.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

function sqliteAvailable(): boolean {
  try {
    createRequire(import.meta.url)("node:sqlite");
    return true;
  } catch {
    return false;
  }
}

async function main(): Promise<void> {
  if (!sqliteAvailable()) {
    log("Skipped: node:sqlite unavailable on this runtime.");
    return;
  }

  // Case 1: directory resolution.
  const work = mkdtempSync(join(tmpdir(), "rath-cfg-it-"));
  const savedOverride = process.env.RATH_CONFIG_DIR;
  process.env.RATH_CONFIG_DIR = work;
  assert.equal(configDir(), work, "$RATH_CONFIG_DIR overrides the config dir");
  delete process.env.RATH_CONFIG_DIR;
  const defaultDir = configDir();
  assert.ok(defaultDir.endsWith("rath"), `default config dir ends in "rath": ${defaultDir}`);
  if (process.platform === "darwin") {
    assert.ok(
      defaultDir.includes(join("Library", "Application Support")),
      "macOS uses Application Support",
    );
  }
  process.env.RATH_CONFIG_DIR = work;
  log("Case 1 OK: configDir honors $RATH_CONFIG_DIR and the default is platform-correct");

  // Case 2: empty store, then write + update + clear.
  assert.deepEqual(loadPreferences(), {}, "fresh store is empty");
  setDefaultModel("openai-native/gpt-5-mini");
  assert.equal(loadPreferences().defaultModel, "openai-native/gpt-5-mini");
  setDefaultModel("openrouter-native/openai/gpt-4o");
  assert.equal(loadPreferences().defaultModel, "openrouter-native/openai/gpt-4o", "update wins");
  clearDefaultModel();
  assert.equal(loadPreferences().defaultModel, undefined, "clear removes the pin");
  setDefaultModel("openrouter-native/openai/gpt-4o"); // restore for later cases
  log("Case 2 OK: round-trip set/update/clear through a real DB file");

  // Case 3: schema migrated automatically, re-open idempotent.
  const { DatabaseSync } = createRequire(import.meta.url)("node:sqlite");
  const dbFile = join(work, "rath.sqlite");
  // Current migration count (v1 preferences, v2 cache); bump as migrations are
  // added.
  const SCHEMA_VERSION = 2;
  {
    const db = new DatabaseSync(dbFile);
    const version = db.prepare("PRAGMA user_version").get().user_version;
    assert.equal(version, SCHEMA_VERSION, "user_version reflects the applied migration count");
    const tables = db
      .prepare(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('preferences','cache')",
      )
      .all()
      .map((r: { name: string }) => r.name)
      .sort();
    assert.deepEqual(tables, ["cache", "preferences"], "both migrated tables exist");
    db.close();
  }
  // Re-open via the store (runs migrate again) — must not error or drift.
  assert.equal(loadPreferences().defaultModel, "openrouter-native/openai/gpt-4o");
  {
    const db = new DatabaseSync(dbFile);
    assert.equal(
      db.prepare("PRAGMA user_version").get().user_version,
      SCHEMA_VERSION,
      "no version drift",
    );
    db.close();
  }
  log(`Case 3 OK: auto-migrated to v${SCHEMA_VERSION}, re-open idempotent`);

  // Case 4: a DB ahead of this build is left alone (no downgrade).
  {
    const db = new DatabaseSync(dbFile);
    db.exec("PRAGMA user_version = 999");
    db.close();
  }
  setDefaultModel("openai-native/gpt-5"); // opens, migrate() should no-op
  {
    const db = new DatabaseSync(dbFile);
    assert.equal(db.prepare("PRAGMA user_version").get().user_version, 999, "future version kept");
    db.close();
  }
  assert.equal(loadPreferences().defaultModel, "openai-native/gpt-5", "still writes/reads at v999");
  log("Case 4 OK: forward-compatible — never downgrades a newer schema");

  if (savedOverride === undefined) {
    delete process.env.RATH_CONFIG_DIR;
  } else {
    process.env.RATH_CONFIG_DIR = savedOverride;
  }
  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
