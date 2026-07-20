export type LLMProvider = "anthropic" | "openai" | "mistral" | "custom";

export interface LLMConfig {
  /** Provider to call. Use "custom" with `adapter` or `baseUrl` for anything else. */
  provider: LLMProvider;
  /**
   * API key for the provider. Optional — omit it when calling a backend proxy /
   * gateway (via `baseUrl`) that injects auth, or when using a custom `adapter`.
   */
  apiKey?: string;
  /** Override the default model for the provider. See `DEFAULT_MODELS`. */
  model?: string;
  /** Max tokens for the response. @default 2048 */
  maxTokens?: number;
  /** Sampling temperature. Left to the provider default when omitted. */
  temperature?: number;
  /**
   * Override the API base URL. Point this at your backend proxy or an
   * OpenAI-compatible gateway (LiteLLM, Azure OpenAI, etc.) to keep API keys
   * off the client. Required for provider "custom" unless `adapter` is set.
   */
  baseUrl?: string;
  /** Extra headers merged into the request (e.g. proxy auth). */
  headers?: Record<string, string>;
  /** Per-request timeout in milliseconds. @default 60000 */
  timeoutMs?: number;
  /** Max retry attempts on transient errors (429 / 5xx / timeout). @default 2 */
  maxRetries?: number;
  /**
   * Fully custom adapter — required when `provider` is "custom" and no `baseUrl`
   * is given. Receives the prompt and (optional) system prompt, returns text.
   */
  adapter?: (prompt: string, systemPrompt?: string) => Promise<string>;
}

export interface TokenUsage {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
}

export interface LLMResult {
  /** The generated text. */
  text: string;
  /** The model that actually served the request (as reported by the provider). */
  model: string;
  /** Token usage, when the provider reports it. */
  usage?: TokenUsage;
  /** Provider stop/finish reason, when available. */
  finishReason?: string;
  /** The raw, unparsed provider response — for debugging or advanced use. */
  raw?: unknown;
}

export interface CompleteRequest {
  /** The user prompt. */
  prompt: string;
  /** Optional system prompt. */
  system?: string;
  /** Override `config.maxTokens` for this call. */
  maxTokens?: number;
  /** Override `config.temperature` for this call. */
  temperature?: number;
  /** Abort signal to cancel the request (and stop retries). */
  signal?: AbortSignal;
  /** Ask the provider for JSON output where supported (OpenAI-compatible `response_format`). */
  responseFormat?: "json" | "text";
}

export interface StreamRequest extends CompleteRequest {
  /** Called for each text delta as it arrives, with the delta and the full accumulated text so far. */
  onToken?: (delta: string, accumulated: string) => void;
}

/** Error thrown for provider/transport failures, carrying HTTP status + provider. */
export class LLMError extends Error {
  readonly status?: number;
  readonly provider?: string;
  constructor(message: string, opts?: { status?: number; provider?: string; cause?: unknown }) {
    super(message);
    this.name = "LLMError";
    this.status = opts?.status;
    this.provider = opts?.provider;
    if (opts?.cause !== undefined) (this as { cause?: unknown }).cause = opts.cause;
  }
}
