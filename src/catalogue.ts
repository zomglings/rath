/**
 * Live model catalogue for rath's native providers.
 *
 * pi-ai ships a bundled model registry, but it goes stale between releases (it
 * missed anthropic/claude-fable-5 for a while). This module fetches the
 * providers' own model lists at runtime so the catalogue stays current without
 * waiting on a pi-ai bump:
 *
 * - OpenRouter: GET /api/v1/models — keyless, rich (id, name, pricing, context).
 * - OpenAI: GET /v1/models — needs the API key; ids only.
 *
 * Results are cached in the SQLite config store with a freshness window, so a
 * fetch happens at most once per window rather than on every run, and a stale
 * cache (or the bundled registry) is used when the network is unavailable.
 * ensureCatalogue() primes the in-memory caches; the accessors are synchronous
 * so the providers and listModels can consult them during model resolution.
 */
import { getCacheEntry, setCacheEntry } from "./config.js";

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const OPENAI_MODELS_URL = "https://api.openai.com/v1/models";

const CACHE_KEY_OPENROUTER = "catalogue:openrouter:v1";
const CACHE_KEY_OPENAI = "catalogue:openai:v1";

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

// In-memory caches, populated by ensureCatalogue. undefined = "not loaded";
// callers fall back to pi-ai's bundled registry in that case.
let openRouterModels: Map<string, RawOpenRouterModel> | undefined;
let openAiModelIds: string[] | undefined;

/** The live OpenRouter catalogue (id -> raw model), or undefined if not loaded. */
export function openRouterCatalogue(): Map<string, RawOpenRouterModel> | undefined {
  return openRouterModels;
}

/** The live OpenAI model ids, or undefined if not loaded (no key / not fetched). */
export function openAiCatalogue(): string[] | undefined {
  return openAiModelIds;
}

/** Drop the in-memory caches (tests). Does not touch the persisted cache. */
export function resetCatalogue(): void {
  openRouterModels = undefined;
  openAiModelIds = undefined;
}

async function fetchJson(url: string, timeoutMs: number, apiKey?: string): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: apiKey ? { authorization: `Bearer ${apiKey}` } : undefined,
    });
    if (!response.ok) {
      throw new Error(`${url} -> HTTP ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Ensure the in-memory catalogues are loaded, fetching from the network only
 * when the cached copy is missing or older than maxAgeMs. Best-effort: a fetch
 * failure falls back to the (possibly stale) cache, and anything missing leaves
 * that provider's catalogue undefined so callers use the bundled registry.
 */
export async function ensureCatalogue(
  opts: { openaiKey?: string; maxAgeMs?: number; timeoutMs?: number; now?: number } = {},
): Promise<void> {
  const maxAgeMs = opts.maxAgeMs ?? DEFAULT_MAX_AGE_MS;
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const now = opts.now ?? Date.now();

  // OpenRouter (keyless).
  if (openRouterModels === undefined) {
    const cached = getCacheEntry(CACHE_KEY_OPENROUTER);
    const fresh = cached && now - cached.fetchedAt < maxAgeMs;
    if (fresh) {
      openRouterModels = indexOpenRouter(safeParseArray(cached.value));
    } else {
      try {
        const payload = (await fetchJson(OPENROUTER_MODELS_URL, timeoutMs)) as {
          data?: RawOpenRouterModel[];
        };
        const models = Array.isArray(payload.data) ? payload.data : [];
        if (models.length > 0) {
          setCacheEntry(CACHE_KEY_OPENROUTER, JSON.stringify(models), now);
          openRouterModels = indexOpenRouter(models);
        } else if (cached) {
          openRouterModels = indexOpenRouter(safeParseArray(cached.value));
        }
      } catch {
        if (cached) {
          openRouterModels = indexOpenRouter(safeParseArray(cached.value));
        }
      }
    }
  }

  // OpenAI (needs the key; ids only).
  if (openAiModelIds === undefined && opts.openaiKey) {
    const cached = getCacheEntry(CACHE_KEY_OPENAI);
    const fresh = cached && now - cached.fetchedAt < maxAgeMs;
    if (fresh) {
      openAiModelIds = safeParseStringArray(cached.value);
    } else {
      try {
        const payload = (await fetchJson(OPENAI_MODELS_URL, timeoutMs, opts.openaiKey)) as {
          data?: { id?: string }[];
        };
        const ids = (Array.isArray(payload.data) ? payload.data : [])
          .map((m) => m.id)
          .filter((id): id is string => typeof id === "string")
          .sort();
        if (ids.length > 0) {
          setCacheEntry(CACHE_KEY_OPENAI, JSON.stringify(ids), now);
          openAiModelIds = ids;
        } else if (cached) {
          openAiModelIds = safeParseStringArray(cached.value);
        }
      } catch {
        if (cached) {
          openAiModelIds = safeParseStringArray(cached.value);
        }
      }
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

function safeParseStringArray(json: string): string[] {
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? parsed.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}
