import type { LLMConfig } from "../types";

/**
 * Anthropic adapter — calls /v1/messages
 */
async function anthropicAdapter(prompt: string, config: LLMConfig): Promise<string> {
  const model = config.model ?? "claude-sonnet-4-20250514";
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
      "x-api-key": config.apiKey,
    },
    body: JSON.stringify({
      model,
      max_tokens: config.maxTokens ?? 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `Anthropic error ${response.status}`);
  return data.content?.find((b: { type: string }) => b.type === "text")?.text ?? "";
}

/**
 * OpenAI adapter — calls /v1/chat/completions (also compatible with Azure OpenAI)
 */
async function openaiAdapter(prompt: string, config: LLMConfig): Promise<string> {
  const model = config.model ?? "gpt-4o";
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model,
      max_tokens: config.maxTokens ?? 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `OpenAI error ${response.status}`);
  return data.choices?.[0]?.message?.content ?? "";
}

/**
 * Mistral adapter — calls /v1/chat/completions
 */
async function mistralAdapter(prompt: string, config: LLMConfig): Promise<string> {
  const model = config.model ?? "mistral-medium";
  const response = await fetch("https://api.mistral.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model,
      max_tokens: config.maxTokens ?? 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `Mistral error ${response.status}`);
  return data.choices?.[0]?.message?.content ?? "";
}

/**
 * Resolves and calls the correct adapter based on config.provider.
 */
export async function callLLM(prompt: string, config: LLMConfig): Promise<string> {
  if (config.provider === "custom") {
    if (!config.adapter) throw new Error("A custom adapter function is required when provider is 'custom'.");
    return config.adapter(prompt);
  }

  const adapters: Record<string, (p: string, c: LLMConfig) => Promise<string>> = {
    anthropic: anthropicAdapter,
    openai:    openaiAdapter,
    mistral:   mistralAdapter,
  };

  const fn = adapters[config.provider];
  if (!fn) throw new Error(`Unknown provider: "${config.provider}". Use anthropic, openai, mistral, or custom.`);
  return fn(prompt, config);
}
