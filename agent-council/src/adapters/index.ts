import type { LLMConfig } from "../types";

async function anthropicAdapter(prompt: string, systemPrompt: string, config: LLMConfig): Promise<string> {
  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
      "x-api-key": config.apiKey,
    },
    body: JSON.stringify({
      model: config.model ?? "claude-sonnet-4-20250514",
      max_tokens: config.maxTokens ?? 1000,
      system: systemPrompt,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `Anthropic error ${response.status}`);
  return data.content?.find((b: { type: string }) => b.type === "text")?.text ?? "";
}

async function openaiAdapter(prompt: string, systemPrompt: string, config: LLMConfig): Promise<string> {
  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model ?? "gpt-4o",
      max_tokens: config.maxTokens ?? 1000,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `OpenAI error ${response.status}`);
  return data.choices?.[0]?.message?.content ?? "";
}

async function mistralAdapter(prompt: string, systemPrompt: string, config: LLMConfig): Promise<string> {
  const response = await fetch("https://api.mistral.ai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model ?? "mistral-medium",
      max_tokens: config.maxTokens ?? 1000,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: prompt },
      ],
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message ?? `Mistral error ${response.status}`);
  return data.choices?.[0]?.message?.content ?? "";
}

export async function callLLM(
  prompt: string,
  systemPrompt: string,
  config: LLMConfig
): Promise<string> {
  if (config.provider === "custom") {
    if (!config.adapter) throw new Error("A custom adapter function is required when provider is 'custom'.");
    return config.adapter(prompt, systemPrompt);
  }
  const adapters: Record<string, (p: string, s: string, c: LLMConfig) => Promise<string>> = {
    anthropic: anthropicAdapter,
    openai:    openaiAdapter,
    mistral:   mistralAdapter,
  };
  const fn = adapters[config.provider];
  if (!fn) throw new Error(`Unknown provider: "${config.provider}".`);
  return fn(prompt, systemPrompt, config);
}
