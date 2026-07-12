export type LLMProvider = "anthropic" | "openai" | "mistral" | "custom";

export interface LLMConfig {
  provider: LLMProvider;
  apiKey?: string;
  model?: string;
  baseUrl?: string;
  headers?: Record<string, string>;
}

export interface LLMAdapter {
  complete(prompt: string, config: LLMConfig): Promise<string>;
}

const DEFAULT_MODELS: Record<string, string> = {
  anthropic: "claude-sonnet-4-20250514",
  openai: "gpt-4o",
  mistral: "mistral-medium",
};

async function readJsonOrThrow(res: Response, provider: string) {
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${provider} API error ${res.status}: ${body.slice(0, 300)}`);
  }
  return res.json();
}

const anthropicAdapter: LLMAdapter = {
  async complete(prompt, config) {
    const res = await fetch(config.baseUrl ?? "https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": config.apiKey ?? "",
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
        ...config.headers,
      },
      body: JSON.stringify({
        model: config.model ?? DEFAULT_MODELS.anthropic,
        max_tokens: 1500,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    const data = await readJsonOrThrow(res, "Anthropic");
    return (data.content ?? [])
      .filter((b: { type: string }) => b.type === "text")
      .map((b: { text: string }) => b.text)
      .join("\n");
  },
};

/** OpenAI-compatible Chat Completions — also used for Mistral and custom gateways (LiteLLM, etc.). */
function chatCompletionsAdapter(defaultUrl: string, providerLabel: string): LLMAdapter {
  return {
    async complete(prompt, config) {
      const res = await fetch(config.baseUrl ?? defaultUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiKey ?? ""}`,
          ...config.headers,
        },
        body: JSON.stringify({
          model: config.model ?? DEFAULT_MODELS[config.provider] ?? config.model,
          messages: [{ role: "user", content: prompt }],
        }),
      });
      const data = await readJsonOrThrow(res, providerLabel);
      return data.choices?.[0]?.message?.content ?? "";
    },
  };
}

export function getAdapter(config: LLMConfig): LLMAdapter {
  switch (config.provider) {
    case "anthropic":
      return anthropicAdapter;
    case "openai":
      return chatCompletionsAdapter("https://api.openai.com/v1/chat/completions", "OpenAI");
    case "mistral":
      return chatCompletionsAdapter("https://api.mistral.ai/v1/chat/completions", "Mistral");
    case "custom":
      if (!config.baseUrl) {
        throw new Error('provider "custom" requires a baseUrl (e.g. your LiteLLM gateway /chat/completions endpoint)');
      }
      return chatCompletionsAdapter(config.baseUrl, "Custom LLM");
    default:
      throw new Error(`Unknown provider: ${(config as LLMConfig).provider}`);
  }
}
