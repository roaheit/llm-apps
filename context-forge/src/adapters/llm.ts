import type { LLMConfig } from "../types";

async function anthropic(prompt: string, system: string, cfg: LLMConfig): Promise<string> {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
      "x-api-key": cfg.apiKey,
    },
    body: JSON.stringify({
      model: cfg.model ?? "claude-sonnet-4-20250514",
      max_tokens: cfg.maxTokens ?? 1500,
      system,
      messages: [{ role: "user", content: prompt }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message ?? `Anthropic ${res.status}`);
  return data.content?.find((b: { type: string }) => b.type === "text")?.text ?? "";
}

async function openai(prompt: string, system: string, cfg: LLMConfig): Promise<string> {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.apiKey}` },
    body: JSON.stringify({
      model: cfg.model ?? "gpt-4o",
      max_tokens: cfg.maxTokens ?? 1500,
      temperature: cfg.temperature ?? 0.2,
      messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message ?? `OpenAI ${res.status}`);
  return data.choices?.[0]?.message?.content ?? "";
}

async function mistral(prompt: string, system: string, cfg: LLMConfig): Promise<string> {
  const res = await fetch("https://api.mistral.ai/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.apiKey}` },
    body: JSON.stringify({
      model: cfg.model ?? "mistral-medium",
      max_tokens: cfg.maxTokens ?? 1500,
      messages: [{ role: "system", content: system }, { role: "user", content: prompt }],
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message ?? `Mistral ${res.status}`);
  return data.choices?.[0]?.message?.content ?? "";
}

export async function callLLM(
  prompt: string,
  systemPrompt: string,
  cfg: LLMConfig
): Promise<string> {
  if (cfg.provider === "custom") {
    if (!cfg.adapter) throw new Error("Custom LLM adapter required.");
    return cfg.adapter(prompt, systemPrompt);
  }
  const map: Record<string, typeof anthropic> = { anthropic, openai, mistral };
  const fn = map[cfg.provider];
  if (!fn) throw new Error(`Unknown LLM provider: "${cfg.provider}"`);
  return fn(prompt, systemPrompt, cfg);
}
