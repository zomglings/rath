/**
 * Model and reasoning selection, shared by every rath command and agent.
 *
 * A model is always named explicitly as <provider>/<model-id>. rath's native
 * providers (openai-native, openrouter-native) resolve through their own
 * constructors — openrouter-native against the live catalogue when primed —
 * and everything else resolves through pi-ai's bundled registry. This lives
 * outside commands/run.ts so subagents (e.g. the Barbarian Reviewer) and
 * future commands can resolve models without dragging in the whole `rath run`
 * frontend.
 */
import {
  type Api,
  getModel,
  getModels,
  getProviders,
  type KnownProvider,
  type Model,
} from "@earendil-works/pi-ai";
import { openRouterCatalogue } from "./catalogue.js";
import { OPENAI_NATIVE_API, openaiNativeModel } from "./providers/openai-native.js";
import { OPENROUTER_NATIVE_API, openrouterNativeModel } from "./providers/openrouter-native.js";

export const REASONING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"] as const;
export type ReasoningLevel = (typeof REASONING_LEVELS)[number];

// The default for the (DB-stored) default model: used when neither -m nor a
// pinned default model (preferences, set via /config default-model) supplies
// one. Hence the doubled name — it is the default of the default.
export const DEFAULT_DEFAULT_MODEL = `${OPENAI_NATIVE_API}/gpt-5.5`;

export function resolveModel(spec: string): Model<Api> {
  const slash = spec.indexOf("/");
  if (slash <= 0 || slash === spec.length - 1) {
    throw new Error(`Model must be <provider>/<model-id> (got: ${spec})`);
  }
  const provider = spec.slice(0, slash);
  const modelId = spec.slice(slash + 1);
  if (provider === OPENAI_NATIVE_API) {
    return openaiNativeModel(modelId);
  }
  if (provider === OPENROUTER_NATIVE_API) {
    return openrouterNativeModel(modelId) as Model<Api>;
  }
  const model = getModel(provider as KnownProvider, modelId as never) as Model<Api> | undefined;
  if (!model) {
    throw new Error(`Unknown model: ${spec}`);
  }
  return model;
}

/**
 * All selectable model specs, native providers first. openrouter-native comes
 * from the live catalogue (ensureCatalogue) when it has been primed, falling
 * back to pi-ai's bundled registry otherwise; openai-native and the stock
 * providers come from the bundled registry (what they validate against).
 */
export function listModels(filter?: string): string[] {
  const specs: string[] = [];
  for (const model of getModels("openai")) {
    specs.push(`${OPENAI_NATIVE_API}/${model.id}`);
  }
  const liveOpenRouter = openRouterCatalogue();
  const openRouterIds = liveOpenRouter
    ? [...liveOpenRouter.keys()].sort()
    : getModels("openrouter").map((m) => m.id);
  for (const id of openRouterIds) {
    specs.push(`${OPENROUTER_NATIVE_API}/${id}`);
  }
  for (const provider of getProviders()) {
    for (const model of getModels(provider)) {
      specs.push(`${provider}/${model.id}`);
    }
  }
  return filter ? specs.filter((s) => s.includes(filter)) : specs;
}
