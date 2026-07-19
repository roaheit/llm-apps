import { assertOk, fetchWithRetry } from "./http";
import { DEFAULT_MODELS } from "./models";
import type { LLMConfig, LLMResult } from "./types";

/** A fully-resolved request — defaults already applied by `complete()`. */
export interface ResolvedRequest {
  prompt: string;
  system: string;
  maxTokens: number;
  temperature?: number;
  timeoutMs: number;
  maxRetries: number;
  signal?: AbortSignal;
}

// The provider JSON payloads are untyped at the boundary; parse defensively.
/* eslint-disable @typescript-eslint/no-explicit-any */
async function readOrThrow(res: Response, provider: string): Promise<any> {
  await assertOk(res, provider);
  return res.json();
}

/** Anthropic Messages API (`/v1/messages`). */
export async function anthropicComplete(config: LLMConfig, req: ResolvedRequest): Promise<LLMResult> {
  const model = config.model ?? DEFAULT_MODELS.anthropic;
  const url = config.baseUrl ?? "https://api.anthropic.com/v1/messages";

  const body: Record<string, unknown> = {
    model,
    max_tokens: req.maxTokens,
    messages: [{ role: "user", content: req.prompt }],
  };
  if (req.system) body.system = req.system;
  if (req.temperature !== undefined) body.temperature = req.temperature;

  const res = await fetchWithRetry(
    url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": config.apiKey ?? "",
        "anthropic-version": "2023-06-01",
        "anthropic-dangerous-direct-browser-access": "true",
        ...config.headers,
      },
      body: JSON.stringify(body),
    },
    { timeoutMs: req.timeoutMs, maxRetries: req.maxRetries, signal: req.signal, provider: "Anthropic" }
  );

  const data = await readOrThrow(res, "Anthropic");
  const text: string = (data.content ?? [])
    .filter((b: { type: string }) => b.type === "text")
    .map((b: { text: string }) => b.text)
    .join("\n");

  return {
    text,
    model: data.model ?? model,
    finishReason: data.stop_reason,
    usage: data.usage
      ? {
          inputTokens: data.usage.input_tokens,
          outputTokens: data.usage.output_tokens,
          totalTokens: (data.usage.input_tokens ?? 0) + (data.usage.output_tokens ?? 0),
        }
      : undefined,
    raw: data,
  };
}

/**
 * OpenAI-compatible Chat Completions — used for OpenAI, Mistral, and any custom
 * gateway (LiteLLM, Azure OpenAI, self-hosted, etc.).
 */
export async function chatComplete(
  config: LLMConfig,
  req: ResolvedRequest,
  opts: { url: string; provider: string; defaultModel?: string }
): Promise<LLMResult> {
  const model = config.model ?? opts.defaultModel;

  const messages: Array<{ role: string; content: string }> = [];
  if (req.system) messages.push({ role: "system", content: req.system });
  messages.push({ role: "user", content: req.prompt });

  const body: Record<string, unknown> = { max_tokens: req.maxTokens, messages };
  if (model) body.model = model;
  if (req.temperature !== undefined) body.temperature = req.temperature;

  const res = await fetchWithRetry(
    opts.url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${config.apiKey ?? ""}`,
        ...config.headers,
      },
      body: JSON.stringify(body),
    },
    { timeoutMs: req.timeoutMs, maxRetries: req.maxRetries, signal: req.signal, provider: opts.provider }
  );

  const data = await readOrThrow(res, opts.provider);
  const choice = data.choices?.[0];

  return {
    text: choice?.message?.content ?? "",
    model: data.model ?? model ?? "",
    finishReason: choice?.finish_reason,
    usage: data.usage
      ? {
          inputTokens: data.usage.prompt_tokens,
          outputTokens: data.usage.completion_tokens,
          totalTokens: data.usage.total_tokens,
        }
      : undefined,
    raw: data,
  };
}
/* eslint-enable @typescript-eslint/no-explicit-any */
