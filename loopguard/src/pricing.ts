import type { TokenUsage } from "corellm";

export interface ModelPricing {
  /** USD per 1,000,000 input tokens. */
  inputPerMTokUsd: number;
  /** USD per 1,000,000 output tokens. */
  outputPerMTokUsd: number;
}

/** Keyed as `"<provider>:<model>"`, with `"<provider>:default"` as a per-provider fallback. */
export type PricingTable = Record<string, ModelPricing>;

/**
 * Illustrative defaults — verify against each provider's current pricing page
 * before relying on cost budgets for real billing decisions. This table is a
 * plain, fully overridable object (mirrors corellm's own DEFAULT_MODELS
 * single-source-of-truth pattern) so it can be updated without touching guard
 * internals.
 */
export const DEFAULT_PRICING: PricingTable = {
  "anthropic:default": { inputPerMTokUsd: 3, outputPerMTokUsd: 15 },
  "openai:default": { inputPerMTokUsd: 2.5, outputPerMTokUsd: 10 },
  "mistral:default": { inputPerMTokUsd: 2, outputPerMTokUsd: 6 },
};

export function resolvePricing(
  table: PricingTable,
  provider?: string,
  model?: string
): ModelPricing | undefined {
  if (provider && model) {
    const exact = table[`${provider}:${model}`];
    if (exact) return exact;
  }
  if (provider) {
    const providerDefault = table[`${provider}:default`];
    if (providerDefault) return providerDefault;
  }
  return undefined;
}

export function estimateCostUsd(
  usage: TokenUsage | undefined,
  pricing: ModelPricing | undefined
): { usd: number; approximate: boolean } | undefined {
  if (!usage || !pricing) return undefined;
  if (usage.inputTokens == null && usage.outputTokens == null) return undefined;

  const inputTokens = usage.inputTokens ?? 0;
  const outputTokens = usage.outputTokens ?? 0;
  const usd =
    (inputTokens / 1_000_000) * pricing.inputPerMTokUsd +
    (outputTokens / 1_000_000) * pricing.outputPerMTokUsd;
  const approximate = usage.inputTokens == null || usage.outputTokens == null;
  return { usd, approximate };
}
