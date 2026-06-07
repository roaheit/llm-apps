# json-storyteller

> A React component that transforms raw JSON into human-readable narratives — works with **any LLM provider**.

Part of [ilm-apps](https://github.com/ilm-apps) — a collection of useful, opinionated React tools.

---

## Installation

```bash
npm install json-storyteller
# or
yarn add json-storyteller
```

---

## Quick Start

```tsx
import { JsonStoryteller } from "json-storyteller";

// Anthropic
<JsonStoryteller
  llm={{ provider: "anthropic", apiKey: "sk-ant-..." }}
  data={{ user: { name: "Ada", plan: "pro", logins: 312 } }}
  tone="narrative"
/>

// OpenAI
<JsonStoryteller
  llm={{ provider: "openai", apiKey: "sk-...", model: "gpt-4o" }}
  data={myData}
  tone="analyst"
/>

// Mistral
<JsonStoryteller
  llm={{ provider: "mistral", apiKey: "...", model: "mistral-medium" }}
  data={myData}
  tone="casual"
/>
```

---

## Supported Providers

| Provider | `provider` value | Default model |
|---|---|---|
| Anthropic | `"anthropic"` | `claude-sonnet-4-20250514` |
| OpenAI | `"openai"` | `gpt-4o` |
| Mistral | `"mistral"` | `mistral-medium` |
| Any other | `"custom"` | — (bring your own adapter) |

---

## API Key Setup

How you manage your API key depends on your use case.

### Local development & prototyping

Store the key in a `.env` file at your project root. Use the prefix for your framework:

```bash
# Vite
VITE_ANTHROPIC_KEY=sk-ant-...

# Next.js
NEXT_PUBLIC_ANTHROPIC_KEY=sk-ant-...

# Create React App
REACT_APP_ANTHROPIC_KEY=sk-ant-...
```

Then pass it in:

```tsx
// Vite
<JsonStoryteller
  llm={{ provider: "anthropic", apiKey: import.meta.env.VITE_ANTHROPIC_KEY }}
  data={myData}
/>

// Next.js
<JsonStoryteller
  llm={{ provider: "anthropic", apiKey: process.env.NEXT_PUBLIC_ANTHROPIC_KEY }}
  data={myData}
/>
```

Always add `.env` to your `.gitignore` — never commit API keys to source control:

```bash
# .gitignore
.env
.env.local
.env*.local
```

### Production apps — use a backend proxy

**Direct browser API calls expose your key in network requests.** Anyone can open DevTools and read it. For any public-facing app, route requests through your own backend instead:

```
Browser → Your backend endpoint → LLM provider
```

Use the `custom` adapter to point to your own endpoint:

```tsx
<JsonStoryteller
  llm={{
    provider: "custom",
    apiKey: "",
    adapter: async (prompt) => {
      const res = await fetch("/api/narrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      return data.story;
    },
  }}
  data={myData}
/>
```

Your backend handles the actual LLM call and keeps the key server-side:

```ts
// pages/api/narrate.ts (Next.js example)
export default async function handler(req, res) {
  const { prompt } = req.body;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
      "x-api-key": process.env.ANTHROPIC_KEY, // server-side only
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1000,
      messages: [{ role: "user", content: prompt }],
    }),
  });

  const data = await response.json();
  const story = data.content?.find(b => b.type === "text")?.text ?? "";
  res.status(200).json({ story });
}
```

### Summary

| Use case | Recommendation |
|---|---|
| Local dev / prototyping | `.env` file, never committed |
| Personal or internal tool | `.env` with framework prefix |
| Public-facing production app | Backend proxy — always |

---

## LLMConfig

| Field | Type | Description |
|---|---|---|
| `provider` | `"anthropic" \| "openai" \| "mistral" \| "custom"` | The LLM provider to use. |
| `apiKey` | `string` | API key for the provider. |
| `model` | `string` | Model override. Falls back to provider default. |
| `maxTokens` | `number` | Max tokens. Default: `1000`. |
| `adapter` | `(prompt: string) => Promise<string>` | Custom adapter (required when provider is `"custom"`). |

---

## Custom Provider

Use `"custom"` to plug in any LLM not built-in — Gemini, Cohere, Ollama, a local model, or your own backend:

```tsx
<JsonStoryteller
  llm={{
    provider: "custom",
    apiKey: "",
    adapter: async (prompt) => {
      const res = await fetch("https://api.your-llm.com/generate", {
        method: "POST",
        headers: { "Authorization": "Bearer YOUR_KEY" },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      return data.output;
    },
  }}
  data={myData}
/>
```

---

## Props

| Prop | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLMConfig` | — | **Required.** LLM provider config. |
| `data` | `object \| string` | — | JSON to narrate. |
| `tone` | `StoryTone` | `"narrative"` | Storytelling style. |
| `onStoryGenerated` | `(story, tone) => void` | — | Callback when story is ready. |
| `onError` | `(error) => void` | — | Callback on error. |
| `headless` | `boolean` | `false` | Render only the story output, no editor UI. |
| `theme` | `"dark" \| "light"` | `"dark"` | Color theme. |
| `className` | `string` | — | Class on the root element. |
| `style` | `CSSProperties` | — | Inline styles on root element. |

### Tones

| Value | Description |
|---|---|
| `"narrative"` | Engaging, story-like prose |
| `"analyst"` | Concise, data-driven report |
| `"casual"` | Conversational, like texting a friend |
| `"poetic"` | Lyrical and metaphorical |

---

## Hook API

```tsx
import { useJsonStoryteller } from "json-storyteller";

const { narrate, story, loading, error, reset } = useJsonStoryteller({
  llm: { provider: "openai", apiKey: import.meta.env.VITE_OPENAI_KEY },
  tone: "casual",
  onStoryGenerated: (story, tone) => console.log(story),
});

<button onClick={() => narrate({ orders: 42, revenue: 8400 })} disabled={loading}>
  {loading ? "Generating…" : "Tell the Story"}
</button>
{story && <p>{story}</p>}
```

### Hook Return Value

| Field | Type | Description |
|---|---|---|
| `narrate` | `(data) => Promise<void>` | Trigger narration with given data. |
| `story` | `string` | The generated narrative. |
| `loading` | `boolean` | True while the API call is in progress. |
| `error` | `Error \| null` | Error from the last call, if any. |
| `reset` | `() => void` | Clear story and error state. |

---

## TypeScript

All types are exported from the package root:

```ts
import type { LLMConfig, LLMProvider, StoryTone, JsonStorytellerProps } from "json-storyteller";
```

---

## License

MIT © [ilm-apps](https://github.com/ilm-apps)
