import { assertOk, backoffDelay, RETRYABLE_STATUS, sleep } from "./http";
import { DEFAULT_MAX_RETRIES, DEFAULT_MAX_TOKENS, DEFAULT_MODELS } from "./models";
import { LLMError, type LLMConfig, type LLMResult, type StreamRequest, type TokenUsage } from "./types";

 

interface ResolvedStream {
  prompt: string;
  system: string;
  maxTokens: number;
  temperature?: number;
  maxRetries: number;
  signal?: AbortSignal;
  responseFormat?: "json" | "text";
  onToken?: (delta: string, accumulated: string) => void;
}

/**
 * Connect for a streaming request, retrying only transient *connect* failures.
 * The external signal is passed straight to `fetch` so it cancels the whole
 * stream; there is no per-request timeout (a stream's duration is unbounded).
 */
async function connectWithRetry(
  url: string,
  init: RequestInit,
  opts: { maxRetries: number; provider: string; signal?: AbortSignal }
): Promise<Response> {
  const { maxRetries, provider, signal } = opts;
  for (let attempt = 0; ; attempt++) {
    if (signal?.aborted) throw signal.reason ?? new LLMError("Request aborted", { provider });
    let res: Response;
    try {
      res = await fetch(url, { ...init, signal });
    } catch (err) {
      if (signal?.aborted) throw signal.reason ?? err;
      if (attempt < maxRetries) {
        await sleep(backoffDelay(attempt), signal);
        continue;
      }
      if (err instanceof LLMError) throw err;
      throw new LLMError(`${provider} request failed: ${(err as Error).message}`, { provider, cause: err });
    }
    if (RETRYABLE_STATUS.has(res.status) && attempt < maxRetries) {
      await sleep(backoffDelay(attempt, Number(res.headers.get("retry-after"))), signal);
      continue;
    }
    return res;
  }
}

/** Parse a `text/event-stream` body into successive `data:` JSON payloads. */
async function* iterSSE(res: Response): AsyncGenerator<any> {
  if (!res.body) return;
  const reader = (res.body as ReadableStream<Uint8Array>).getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line || line.startsWith(":") || line.startsWith("event:")) continue;
        if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          if (data === "[DONE]") return;
          try {
            yield JSON.parse(data);
          } catch {
            /* ignore keep-alives / partial frames */
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function buildUsage(input?: number, output?: number): TokenUsage | undefined {
  if (input == null && output == null) return undefined;
  return { inputTokens: input, outputTokens: output, totalTokens: (input ?? 0) + (output ?? 0) };
}

/** Anthropic Messages streaming (SSE). */
async function anthropicStream(config: LLMConfig, req: ResolvedStream): Promise<LLMResult> {
  const model = config.model ?? DEFAULT_MODELS.anthropic;
  const url = config.baseUrl ?? "https://api.anthropic.com/v1/messages";

  const body: Record<string, unknown> = {
    model,
    max_tokens: req.maxTokens,
    stream: true,
    messages: [{ role: "user", content: req.prompt }],
  };
  if (req.system) body.system = req.system;
  if (req.temperature !== undefined) body.temperature = req.temperature;

  const res = await connectWithRetry(
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
    { maxRetries: req.maxRetries, provider: "Anthropic", signal: req.signal }
  );
  await assertOk(res, "Anthropic");

  let text = "";
  let resolvedModel = model;
  let finishReason: string | undefined;
  let inputTokens: number | undefined;
  let outputTokens: number | undefined;

  for await (const ev of iterSSE(res)) {
    if (ev.type === "message_start") {
      resolvedModel = ev.message?.model ?? resolvedModel;
      inputTokens = ev.message?.usage?.input_tokens ?? inputTokens;
    } else if (ev.type === "content_block_delta") {
      if (ev.delta?.type === "text_delta" && ev.delta.text) {
        text += ev.delta.text;
        req.onToken?.(ev.delta.text, text);
      }
    } else if (ev.type === "message_delta") {
      if (ev.usage?.output_tokens != null) outputTokens = ev.usage.output_tokens;
      if (ev.delta?.stop_reason) finishReason = ev.delta.stop_reason;
    }
  }

  return { text, model: resolvedModel, finishReason, usage: buildUsage(inputTokens, outputTokens) };
}

/** OpenAI-compatible Chat Completions streaming (SSE) — OpenAI, Mistral, custom gateways. */
async function chatStream(
  config: LLMConfig,
  req: ResolvedStream,
  opts: { url: string; provider: string; defaultModel?: string }
): Promise<LLMResult> {
  const model = config.model ?? opts.defaultModel;

  const messages: Array<{ role: string; content: string }> = [];
  if (req.system) messages.push({ role: "system", content: req.system });
  messages.push({ role: "user", content: req.prompt });

  const body: Record<string, unknown> = {
    max_tokens: req.maxTokens,
    messages,
    stream: true,
    // Ask OpenAI-compatible endpoints to include usage in the final chunk.
    stream_options: { include_usage: true },
  };
  if (model) body.model = model;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.responseFormat === "json") body.response_format = { type: "json_object" };

  const res = await connectWithRetry(
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
    { maxRetries: req.maxRetries, provider: opts.provider, signal: req.signal }
  );
  await assertOk(res, opts.provider);

  let text = "";
  let resolvedModel = model ?? "";
  let finishReason: string | undefined;
  let usage: TokenUsage | undefined;

  for await (const ev of iterSSE(res)) {
    const choice = ev.choices?.[0];
    const delta = choice?.delta?.content;
    if (delta) {
      text += delta;
      req.onToken?.(delta, text);
    }
    if (choice?.finish_reason) finishReason = choice.finish_reason;
    if (ev.model) resolvedModel = ev.model;
    if (ev.usage) {
      usage = {
        inputTokens: ev.usage.prompt_tokens,
        outputTokens: ev.usage.completion_tokens,
        totalTokens: ev.usage.total_tokens,
      };
    }
  }

  return { text, model: resolvedModel, finishReason, usage };
}

/**
 * Stream a completion, invoking `request.onToken` for each text delta as it
 * arrives, and resolving to the final LLMResult (with usage) once complete.
 *
 * Providers that can't stream natively (a custom `adapter`) fall back to a
 * single `onToken` call with the full text.
 */
export async function stream(config: LLMConfig, request: StreamRequest): Promise<LLMResult> {
  const req: ResolvedStream = {
    prompt: request.prompt,
    system: request.system ?? "",
    maxTokens: request.maxTokens ?? config.maxTokens ?? DEFAULT_MAX_TOKENS,
    temperature: request.temperature ?? config.temperature,
    maxRetries: config.maxRetries ?? DEFAULT_MAX_RETRIES,
    signal: request.signal,
    responseFormat: request.responseFormat,
    onToken: request.onToken,
  };

  switch (config.provider) {
    case "anthropic":
      return anthropicStream(config, req);

    case "openai":
      return chatStream(config, req, {
        url: config.baseUrl ?? "https://api.openai.com/v1/chat/completions",
        provider: "OpenAI",
        defaultModel: DEFAULT_MODELS.openai,
      });

    case "mistral":
      return chatStream(config, req, {
        url: config.baseUrl ?? "https://api.mistral.ai/v1/chat/completions",
        provider: "Mistral",
        defaultModel: DEFAULT_MODELS.mistral,
      });

    case "custom": {
      if (config.adapter) {
        const text = await config.adapter(req.prompt, req.system || undefined);
        req.onToken?.(text, text);
        return { text, model: config.model ?? "custom" };
      }
      if (config.baseUrl) {
        return chatStream(config, req, { url: config.baseUrl, provider: "Custom LLM" });
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
 
