import type { ReasoningLevel } from "./models.js";

export const DEFAULT_HORDE_CHIEFTAIN_MODEL = "openrouter-native/openai/gpt-5.6-sol";
export const DEFAULT_HORDE_CHIEFTAIN_REASONING: ReasoningLevel = "high";
export const DEFAULT_HORDE_ATTACK_MODEL = "openrouter-native/openai/gpt-5.6-terra";
export const DEFAULT_HORDE_ATTACK_REASONING: ReasoningLevel = "medium";

interface HordeIntelligenceOptions {
  model?: string;
  reasoning?: ReasoningLevel;
  hordeModel?: string;
  hordeReasoning?: ReasoningLevel;
}

export function hordeIntelligence(options: HordeIntelligenceOptions): {
  chieftainModelSpec: string;
  chieftainReasoning: ReasoningLevel;
  hordeModelSpec: string;
  hordeReasoning: ReasoningLevel;
} {
  return {
    chieftainModelSpec: options.model?.trim() || DEFAULT_HORDE_CHIEFTAIN_MODEL,
    chieftainReasoning: options.reasoning ?? DEFAULT_HORDE_CHIEFTAIN_REASONING,
    hordeModelSpec: options.hordeModel?.trim() || DEFAULT_HORDE_ATTACK_MODEL,
    hordeReasoning: options.hordeReasoning ?? DEFAULT_HORDE_ATTACK_REASONING,
  };
}
