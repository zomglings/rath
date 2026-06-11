/**
 * Integration test: the live model catalogue. Hits OpenRouter's keyless
 * /api/v1/models endpoint (no API key, no token cost), so it needs network but
 * not credentials; it skips cleanly when offline. Checks:
 *  1. ensureCatalogue fetches a non-trivial OpenRouter list and persists it to
 *     the SQLite cache.
 *  2. openrouter-native validates against the live list: a live id builds a
 *     Model with numeric per-million costs; an id not in the list is rejected.
 *  3. The cache is reused within the freshness window — a second prime after a
 *     reset loads from the cache without re-fetching (proven with timeoutMs=1).
 *
 * $RATH_CONFIG_DIR is pointed at a temp dir. Exits 0 on success (or clean skip),
 * 1 on failure.
 */
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  ensureCatalogue,
  getCacheEntry,
  openRouterCatalogue,
  openrouterNativeModel,
  registerOpenRouterNative,
  resetCatalogue,
} from "../index.js";

function log(message: string): void {
  process.stdout.write(`${message}\n`);
}

async function main(): Promise<void> {
  process.env.RATH_CONFIG_DIR = mkdtempSync(join(tmpdir(), "rath-catalogue-it-"));
  registerOpenRouterNative();

  const fetchedAt = 1_700_000_000_000; // fixed epoch so freshness is deterministic
  await ensureCatalogue({ timeoutMs: 5000, now: fetchedAt });

  const live = openRouterCatalogue();
  if (!live) {
    log("Skipped: OpenRouter catalogue could not be fetched (offline?).");
    return;
  }

  // Case 1: fetched and persisted.
  assert.ok(live.size > 100, `expected a large catalogue, got ${live.size}`);
  assert.ok(getCacheEntry("catalogue:openrouter:v1"), "catalogue persisted to the SQLite cache");
  log(`Case 1 OK: fetched ${live.size} OpenRouter models and cached them`);

  // Case 2: live validation + model build, and rejection of unknown ids.
  const sampleId =
    [...live.keys()].find((id) => id.startsWith("anthropic/")) ?? [...live.keys()][0];
  assert.ok(sampleId, "catalogue has at least one model");
  const model = openrouterNativeModel(sampleId);
  assert.equal(model.id, sampleId);
  assert.equal(model.api, "openrouter-native");
  assert.equal(typeof model.cost.input, "number");
  assert.equal(typeof model.cost.output, "number");
  assert.ok(model.contextWindow >= 0, "context window is numeric");
  assert.throws(
    () => openrouterNativeModel("nope/not-a-real-model-xyz"),
    /Unknown OpenRouter model/,
    "an id absent from the live list is rejected",
  );
  log(`Case 2 OK: built ${sampleId} from live metadata; unknown id rejected`);

  // Case 3: cache reuse within the freshness window (no network on second prime).
  resetCatalogue();
  assert.equal(openRouterCatalogue(), undefined, "reset cleared the in-memory catalogue");
  // timeoutMs=1 would fail any real fetch; a fresh cache must be used instead.
  await ensureCatalogue({ timeoutMs: 1, now: fetchedAt + 60_000 });
  const reloaded = openRouterCatalogue();
  assert.ok(reloaded && reloaded.size === live.size, "reloaded from cache without re-fetching");
  log("Case 3 OK: cache reused within the freshness window (no re-fetch)");

  log("All assertions passed.");
}

main().then(
  () => process.exit(0),
  (error) => {
    console.error(error instanceof Error ? (error.stack ?? error.message) : error);
    process.exit(1);
  },
);
