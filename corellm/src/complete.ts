import { anthropicComplete, chatComplete } from "./providers";
import { DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOKENS, DEFAULT_MODELS, DEFAULT_TIMEOUT_MS } from "./models";
import { LLMError, type CompleteRequest, type LLMConfig, type LLMResult } from "./types";

/**
 * Call an LLM provider and return the full result (text + usage + metadata).
 * This is the single entry point every llm-apps package builds on.
 */
export async function complete(config: LLMConfig, request: CompleteRequest): Promise<LLMResult> {
  const resolved = {
    prompt: request.prompt,
    system: request.system ?? "",
    maxTokens: request.maxTokens ?? config.maxTokens ?? DEFAULT_MAX_TOKENS,
    temperature: request.temperature ?? config.temperature,
    timeoutMs: config.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    maxRetries: config.maxRetries ?? DEFAULT_MAX_RETRIES,
    signal: request.signal,
    responseFormat: request.responseFormat,
  };

  switch (config.provider) {
    case "anthropic":
      return anthropicComplete(config, resolved);

    case "openai":
      return chatComplete(config, resolved, {
        url: config.baseUrl ?? "https://api.openai.com/v1/chat/completions",
        provider: "OpenAI",
        defaultModel: DEFAULT_MODELS.openai,
      });

    case "mistral":
      return chatComplete(config, resolved, {
        url: config.baseUrl ?? "https://api.mistral.ai/v1/chat/completions",
        provider: "Mistral",
        defaultModel: DEFAULT_MODELS.mistral,
      });

    case "custom": {
      if (config.adapter) {
        const text = await config.adapter(resolved.prompt, resolved.system || undefined);
        return { text, model: config.model ?? "custom" };
      }
      if (config.baseUrl) {
        return chatComplete(config, resolved, { url: config.baseUrl, provider: "Custom LLM" });
      }
      throw new LLMError(
        'provider "custom" requires either an `adapter` function or a `baseUrl` (OpenAI-compatible endpoint).',
        { provider: "custom" }
      );
    }

    default:
      throw new LLMError(`Unknown LLM provider: "${(config as LLMConfig).provider}"`, {
        provider: String((config as LLMConfig).provider),
      });
  }
}

/**
 * Convenience wrapper that returns only the text.
 * Signature-compatible with the previous per-package `callLLM(prompt, system, config)`.
 */
export async function callLLM(
  prompt: string,
  system: string | undefined,
  config: LLMConfig
): Promise<string> {
  const { text } = await complete(config, { prompt, system });
  return text;
}
