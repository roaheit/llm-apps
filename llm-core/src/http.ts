import { LLMError } from "./types";

/** HTTP statuses worth retrying — rate limits, transient 5xx, and Anthropic's 529 "overloaded". */
export const RETRYABLE_STATUS = new Set([408, 425, 429, 500, 502, 503, 504, 529]);

export function backoffDelay(attempt: number, retryAfterSec?: number): number {
  if (retryAfterSec && Number.isFinite(retryAfterSec)) return retryAfterSec * 1000;
  const base = 500 * 2 ** attempt; // 500ms, 1s, 2s, ...
  const jitter = Math.random() * 250;
  return Math.min(base + jitter, 15_000);
}

export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? new Error("Aborted"));
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason ?? new Error("Aborted"));
      },
      { once: true }
    );
  });
}

/** Throw a rich LLMError if the response is not OK, reading the (JSON or text) error body. */
export async function assertOk(res: Response, provider: string): Promise<void> {
  if (res.ok) return;
  const body = await res.text().catch(() => "");
  let message = `${provider} API error ${res.status}`;
  try {
    message = JSON.parse(body)?.error?.message ?? message;
  } catch {
    if (body) message += `: ${body.slice(0, 300)}`;
  }
  throw new LLMError(message, { status: res.status, provider });
}

export interface RetryOptions {
  timeoutMs: number;
  maxRetries: number;
  provider: string;
  signal?: AbortSignal;
}

/**
 * `fetch` with a per-attempt timeout, exponential backoff on transient errors,
 * and cooperative cancellation via an external `AbortSignal`.
 *
 * A user-triggered abort (external signal) is never retried; timeouts and
 * network errors are retried up to `maxRetries`.
 */
export async function fetchWithRetry(
  url: string,
  init: RequestInit,
  opts: RetryOptions
): Promise<Response> {
  const { timeoutMs, maxRetries, provider, signal: external } = opts;

  for (let attempt = 0; ; attempt++) {
    if (external?.aborted) throw external.reason ?? new LLMError("Request aborted", { provider });

    const controller = new AbortController();
    const onExternalAbort = () => controller.abort(external?.reason);
    external?.addEventListener("abort", onExternalAbort, { once: true });
    const timer = setTimeout(
      () => controller.abort(new LLMError(`${provider} request timed out after ${timeoutMs}ms`, { provider })),
      timeoutMs
    );

    try {
      const res = await fetch(url, { ...init, signal: controller.signal });

      if (RETRYABLE_STATUS.has(res.status) && attempt < maxRetries) {
        const retryAfter = Number(res.headers.get("retry-after"));
        await sleep(backoffDelay(attempt, retryAfter), external);
        continue;
      }
      return res;
    } catch (err) {
      // User cancellation — surface it, never retry.
      if (external?.aborted) throw external.reason ?? err;
      // Timeout or network error — retry if attempts remain.
      if (attempt < maxRetries) {
        await sleep(backoffDelay(attempt), external);
        continue;
      }
      if (err instanceof LLMError) throw err;
      throw new LLMError(`${provider} request failed: ${(err as Error).message}`, { provider, cause: err });
    } finally {
      clearTimeout(timer);
      external?.removeEventListener("abort", onExternalAbort);
    }
  }
}
