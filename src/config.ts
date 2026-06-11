/**
 * rath's per-user configuration: a SQLite database under a platform config
 * directory, holding CLI preferences.
 *
 * Directory resolution follows the platformdirs convention clacks uses (a
 * "rath" folder under the OS config base); $RATH_CONFIG_DIR overrides it
 * outright (handy for tests and for relocating the store).
 *
 * Schema is versioned and migrated automatically on open, the lesson taken
 * from clacks: migrations run before any read/write (no manual step), and the
 * MIGRATIONS list is append-only — never edit a shipped entry, only add the
 * next one. PRAGMA user_version is the applied-revision counter (the node:sqlite
 * analogue of clacks's Alembic revision tracking); each pending step runs in a
 * transaction and bumps it, so an existing user's DB upgrades in place.
 *
 * Preferences are CLI-only sugar; the rath library reads nothing implicit. The
 * SQLite module is loaded defensively, so on a runtime without node:sqlite the
 * store degrades to "no persistence" rather than breaking `rath run`.
 */
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { homedir } from "node:os";
import { join } from "node:path";

const APP = "rath";

/**
 * The rath config directory for this user/platform. Does not create it (writes
 * do, lazily). Resolution: $RATH_CONFIG_DIR, else the platform config base —
 * %APPDATA% on Windows, ~/Library/Application Support on macOS,
 * $XDG_CONFIG_HOME or ~/.config elsewhere — joined with "rath".
 */
export function configDir(): string {
  const override = process.env.RATH_CONFIG_DIR?.trim();
  if (override) {
    return override;
  }
  if (process.platform === "win32") {
    const base = process.env.APPDATA || join(homedir(), "AppData", "Roaming");
    return join(base, APP);
  }
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", APP);
  }
  const base = process.env.XDG_CONFIG_HOME?.trim() || join(homedir(), ".config");
  return join(base, APP);
}

function dbPath(): string {
  return join(configDir(), "rath.sqlite");
}

// Minimal shape of the node:sqlite surface we use (we load it via require, so
// it is otherwise untyped). Query rows come back untyped; callers cast.
interface SqliteStatement {
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
  run(...params: unknown[]): unknown;
}
interface SqliteDatabase {
  exec(sql: string): void;
  prepare(sql: string): SqliteStatement;
  close(): void;
}
interface SqliteModule {
  DatabaseSync: new (path: string) => SqliteDatabase;
}

let sqliteModule: SqliteModule | null | undefined;

/** node:sqlite if this runtime has it (Node 22.5+/24), else null (cached). */
function getSqlite(): SqliteModule | null {
  if (sqliteModule === undefined) {
    try {
      sqliteModule = createRequire(import.meta.url)("node:sqlite") as SqliteModule;
    } catch {
      sqliteModule = null;
    }
  }
  return sqliteModule;
}

/**
 * Schema migrations, applied in order. APPEND-ONLY: never edit or remove a
 * shipped entry — only add the next one. The array length is the target schema
 * version; PRAGMA user_version records how many have run on a given database.
 */
const MIGRATIONS: Array<(db: SqliteDatabase) => void> = [
  // v1: preferences as a key/value table (extensible without schema churn).
  (db) => {
    db.exec("CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL)");
  },
];

/** Apply any migrations the database has not yet seen. */
function migrate(db: SqliteDatabase): void {
  const row = db.prepare("PRAGMA user_version").get() as { user_version?: number };
  const applied = Number(row?.user_version ?? 0);
  for (let version = applied; version < MIGRATIONS.length; version++) {
    db.exec("BEGIN");
    try {
      MIGRATIONS[version]?.(db);
      // user_version takes an integer literal (no bind params); version+1 is
      // ours, not user input.
      db.exec(`PRAGMA user_version = ${version + 1}`);
      db.exec("COMMIT");
    } catch (error) {
      db.exec("ROLLBACK");
      throw error;
    }
  }
}

/**
 * Open the config database (creating the directory and file as needed) with
 * the schema migrated to the latest version. Returns null when node:sqlite is
 * unavailable, so callers can degrade to no persistence.
 */
function openDb(): SqliteDatabase | null {
  const sqlite = getSqlite();
  if (!sqlite) {
    return null;
  }
  mkdirSync(configDir(), { recursive: true });
  const db = new sqlite.DatabaseSync(dbPath());
  migrate(db);
  return db;
}

/** Persisted CLI preferences. Optional fields so an empty store is valid. */
export interface Preferences {
  /**
   * Pinned default model (provider/model-id) for new sessions when -m is
   * absent. Set explicitly via `/config default-model`; independent of the
   * model the active session happens to switch to.
   */
  defaultModel?: string;
}

// Preference row keys are snake_case in the database; the Preferences object
// exposes them camelCase. Keep this map as the single place the two meet.
const KEY_DEFAULT_MODEL = "default_model";

/** Read all preferences, returning {} when the store is empty or unavailable. */
export function loadPreferences(): Preferences {
  const db = openDb();
  if (!db) {
    return {};
  }
  try {
    const rows = db.prepare("SELECT key, value FROM preferences").all() as Array<{
      key: string;
      value: string;
    }>;
    const byKey = new Map(rows.map(({ key, value }) => [key, value]));
    const prefs: Preferences = {};
    const defaultModel = byKey.get(KEY_DEFAULT_MODEL);
    if (defaultModel !== undefined) {
      prefs.defaultModel = defaultModel;
    }
    return prefs;
  } catch {
    return {};
  } finally {
    db.close();
  }
}

/** Pin `spec` as the default model for new sessions. Best-effort write. */
export function setDefaultModel(spec: string): void {
  const db = openDb();
  if (!db) {
    return;
  }
  try {
    db.prepare(
      "INSERT INTO preferences (key, value) VALUES (?, ?) " +
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    ).run(KEY_DEFAULT_MODEL, spec);
  } catch {
    // Persistence is best-effort; a failed write must not break the session.
  } finally {
    db.close();
  }
}

/** Clear the pinned default model, reverting new sessions to the built-in. */
export function clearDefaultModel(): void {
  const db = openDb();
  if (!db) {
    return;
  }
  try {
    db.prepare("DELETE FROM preferences WHERE key = ?").run(KEY_DEFAULT_MODEL);
  } catch {
    // Best-effort.
  } finally {
    db.close();
  }
}
