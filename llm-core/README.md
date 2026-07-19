# llm-core

The shared LLM client that powers every package in **llm-apps**. One provider
layer instead of a `callLLM` copy-pasted into each component.

- **Multi-provider** — Anthropic, OpenAI, Mistral, or any OpenAI-compatible
  gateway (LiteLLM, Azure OpenAI, self-hosted) via `provider: "custom"`.
- **Proxy-friendly** — set `baseUrl` (and optional `headers`) to route through a
  backend so API keys never ship to the browser.
- **Streaming** — `stream()` yields text deltas as they arrive (SSE), with a
  graceful one-shot fallback for custom adapters.
- **Resilient** — per-request timeout, exponential backoff with jitter on
  `429`/`5xx`/`529`, `Retry-After` support, and cooperative cancellation via
  `AbortSignal`.
- **Observable** — returns token `usage`, `finishReason`, resolved `model`, and
  the `raw` provider response.
- **Framework-agnostic** — uses `fetch`; no React, no Node-only APIs.

## Usage

```ts
import { complete, callLLM, type LLMConfig } from "llm-core";

const config: LLMConfig = { provider: "anthropic", apiKey: process.env.ANTHROPIC_KEY };

// Full result (text + usage + metadata)
const res = await complete(config, { prompt: "Summarize this...", system: "Be terse." });
console.log(res.text, res.usage, res.model);

// Just the text (drop-in for the old per-package helper)
const text = await callLLM("Summarize this...", "Be terse.", config);
```

### Stream tokens as they arrive

```ts
import { stream } from "llm-core";

const res = await stream(config, {
  prompt: "Write a short story about a robot.",
  onToken: (delta, accumulated) => updateUI(accumulated),
});
console.log(res.usage); // final usage is still returned when the stream ends
```

### Route through a proxy (keep keys off the client)

```ts
const config: LLMConfig = {
  provider: "custom",
  baseUrl: "https://your-app.com/api/llm", // OpenAI-compatible endpoint
  headers: { Authorization: `Bearer ${sessionToken}` },
};
```

## Default models

Defaults live in one place — [`src/models.ts`](src/models.ts) — and every
package inherits them. Override per call with `config.model`.

| Provider  | Default                |
| --------- | ---------------------- |
| anthropic | `claude-sonnet-5`      |
| openai    | `gpt-4o`               |
| mistral   | `mistral-large-latest` |
