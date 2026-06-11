/**
 * Live model catalogue for the openrouter-native provider.
 *
 * pi-ai ships a bundled model registry, but it goes stale between releases (it
 * missed anthropic/claude-fable-5 for a while). OpenRouter exposes a rich,
 * keyless models endpoint (GET /api/v1/models — id, name, pricing, context),
 * so this module fetches it at runtime to keep the openrouter-native catalogue
 * current without waiting on a pi-ai bump.
 *
 * (OpenAI is intentionally not fetched: its /v1/models endpoint is unfiltered —
 * it lists embeddings, audio, and image models alongside chat — and carries no
 * pricing or context metadata, making it a worse source than pi-ai's curated
 * bundled `openai` registry, which the openai-native provider validates
 * against. openai-native therefore stays on the bundled registry.)
 *
 * Results are cached in the SQLite config store with a freshness window, so a
 * fetch happens at most once per window rather than on every run, and a stale
 * cache (or the bundled registry) is used when the network is unavailable.
 * ensureCatalogue() primes the in-memory cache; the accessor is synchronous so
 * the provider and listModels can consult it during model resolution.
 */
import { getCacheEntry, setCacheEntry } from "./config.js";

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const CACHE_KEY_OPENROUTER = "catalogue:openrouter:v1";

const DEFAULT_MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24h
const DEFAULT_TIMEOUT_MS = 2500;

/** The OpenRouter /models fields rath uses; the payload has many more. */
export interface RawOpenRouterModel {
  id: string;
  name?: string;
  context_length?: number;
  architecture?: { input_modalities?: string[] };
  pricing?: {
    prompt?: string;
    completion?: string;
    input_cache_read?: string;
    input_cache_write?: string;
  };
  top_provider?: { context_length?: number; max_completion_tokens?: number | null };
  supported_parameters?: string[];
}

// In-memory cache, populated by ensureCatalogue. undefined = "not loaded";
// callers fall back to pi-ai's bundled registry in that case.
let openRouterModels: Map<string, RawOpenRouterModel> | undefined;

/** The live OpenRouter catalogue (id -> raw model), or undefined if not loaded. */
export function openRouterCatalogue(): Map<string, RawOpenRouterModel> | undefined {
  return openRouterModels;
}

/** Drop the in-memory cache (tests). Does not touch the persisted cache. */
export function resetCatalogue(): void {
  openRouterModels = undefined;
}

async function fetchJson(url: string, timeoutMs: number): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`${url} -> HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ensure the in-memory catalogue is loaded, fetching from the network only when
 * the cached copy is missing or older than maxAgeMs. Best-effort: a fetch
 * failure falls back to the (possibly stale) cache, and anything missing leaves
 * the catalogue undefined so callers use the bundled registry.
 */
export async function ensureCatalogue(
  opts: { maxAgeMs?: number; timeoutMs?: number; now?: number } = {},
): Promise<void> {
  if (openRouterModels !== undefined) {
    return;
  }
  const maxAgeMs = opts.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const now = opts.now ?? Date.now();

  // A cached value that indexes to an EMPTY map (corrupt row, id-less entries,
  // empty array) is treated as "not loaded" so it never poisons the catalogue
  // into rejecting every id — we refetch, then fall back to bundled.
  const cached = getCacheEntry(CACHE_KEY_OPENROUTER);
  const cachedMap = cached ? indexOpenRouter(safeParseArray(cached.value)) : new Map();
  const fresh = cached !== undefined && now - cached.fetchedAt < maxAgeMs;
  if (fresh && cachedMap.size > 0) {
    openRouterModels = cachedMap;
    return;
  }
  try {
    const payload = (await fetchJson(OPENROUTER_MODELS_URL, timeoutMs)) as {
      data?: RawOpenRouterModel[];
    };
    const fetched = indexOpenRouter(Array.isArray(payload.data) ? payload.data : []);
    if (fetched.size > 0) {
      setCacheEntry(CACHE_KEY_OPENROUTER, JSON.stringify([...fetched.values()]), now);
      openRouterModels = fetched;
    } else if (cachedMap.size > 0) {
      openRouterModels = cachedMap;
    }
  } catch {
    if (cachedMap.size > 0) {
      openRouterModels = cachedMap;
    }
  }
}

function indexOpenRouter(models: RawOpenRouterModel[]): Map<string, RawOpenRouterModel> {
  const map = new Map<string, RawOpenRouterModel>();
  for (const model of models) {
    if (model && typeof model.id === "string") {
      map.set(model.id, model);
    }
  }
  return map;
}

function safeParseArray(json: string): RawOpenRouterModel[] {
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}
