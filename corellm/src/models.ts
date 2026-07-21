import type { LLMProvider } from "./types";

/**
 * Default model per provider — the single source of truth for the whole repo.
 * Override per call with `config.model`. Update these here as providers ship
 * new models; every package picks up the change.
 */
export const DEFAULT_MODELS: Record<Exclude<LLMProvider, "custom">, string> = {
  anthropic: "claude-sonnet-5",
  openai: "gpt-4o",
  mistral: "mistral-large-latest",
};

export const DEFAULT_MAX_TOKENS = 2048;
export const DEFAULT_TIMEOUT_MS = 60_000;
export const DEFAULT_MAX_RETRIES = 2;
