# tool-pilot

> Task-planning AI agent with a ReAct reasoning loop and pluggable tools for React. Give it a goal — watch it think, act, and deliver.

Part of [llm-apps](https://github.com/roaheit/llm-apps) — a growing collection of real-world LLM applications & AI agents.

---

## Installation

```bash
npm install tool-pilot
# or
yarn add tool-pilot
```

---

## Quick Start

```tsx
import { ToolPilot, webSearch, fileRead, createCodeExecTool } from "tool-pilot";

// code_exec runs model-generated JS in the host environment (no sandbox) —
// enable it explicitly, and only for trusted input.
const codeExec = createCodeExecTool({ acknowledgeUnsafe: true });

const config = {
  llm: { provider: "anthropic", apiKey: import.meta.env.VITE_ANTHROPIC_KEY },
  tools: [webSearch, fileRead, codeExec],
  maxSteps: 8,
};

export default function App() {
  return (
    <ToolPilot
      config={config}
      placeholder="What should the agent do?"
    />
  );
}
```

That's it. The agent will:

1. **Think** — break the task into steps
2. **Act** — pick a tool and call it
3. **Observe** — read the result
4. **Repeat** until it has a final answer

Every step is streamed into a live visual trace.

---

## How It Works

```
User goal
    │
    ▼
┌──────────────────────────────────────────┐
│           ReAct Reasoning Loop            │
│                                          │
│  THINK → ACTION → OBSERVATION → repeat   │
│  (up to maxSteps iterations)             │
└──────────────────────────────────────────┘
    │
    ▼
Final answer (grounded in tool observations)
```

The agent follows a **ReAct** (Reason + Act) loop:

1. The LLM reasons about what to do next (`THINK`)
2. It decides to use a tool (`ACTION` + `ACTION_INPUT`)
3. The tool executes and returns a result (`OBSERVATION`)
4. The LLM reasons again with the new information
5. When it has enough, it produces a final `ANSWER`

---

## Built-in Tools

### `webSearch`

Searches the web via DuckDuckGo Instant Answer API.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | ✓ | Search query |
| `maxResults` | `number` | | Max results (default: 5) |

### `createCodeExecTool(options)`

Creates a `code_exec` tool that evaluates a JavaScript snippet.

> ⚠️ **Unsafe — no sandbox.** It runs model-generated JavaScript in the host environment via the `Function` constructor; the curated globals are a convenience, not a security boundary. Only enable it for **trusted** input. You must pass `acknowledgeUnsafe: true`; `timeoutMs` (default 5000) is best-effort and cannot interrupt a synchronous infinite loop — use a Web Worker for hard isolation.

```ts
const codeExec = createCodeExecTool({ acknowledgeUnsafe: true, timeoutMs: 5000 });
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `code` | `string` | ✓ | JavaScript code to run |

### `fileRead`

Fetches and returns the text content of an http(s) URL (HTML tags stripped). Subject to CORS in the browser; requests to loopback/private-network addresses are blocked (SSRF guard).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | `string` | ✓ | URL to fetch |
| `maxLength` | `number` | | Max characters (default: 4000) |

---

## Custom Tools

Create any tool by implementing `ToolDefinition`:

```ts
import type { ToolDefinition } from "tool-pilot";

const weatherTool: ToolDefinition = {
  name: "get_weather",
  description: "Get current weather for a city",
  parameters: [
    { name: "city", type: "string", description: "City name", required: true },
  ],
  execute: async (args) => {
    const res = await fetch(`https://wttr.in/${args.city}?format=3`);
    return res.text();
  },
};

// Use it:
const config = {
  llm: { provider: "openai", apiKey: "..." },
  tools: [weatherTool, webSearch],
};
```

The LLM automatically sees all registered tools and their parameter schemas in its system prompt.

---

## Config

```ts
interface ToolPilotConfig {
  llm: LLMConfig;           // LLM provider config
  tools: ToolDefinition[];   // Tools the agent can use
  systemPrompt?: string;     // Custom preamble (replaces the default intro)
  maxSteps?: number;         // Max reasoning steps (default: 10)
}
```

### LLMConfig

| Field | Type | Description |
|---|---|---|
| `provider` | `"anthropic" \| "openai" \| "mistral" \| "custom"` | LLM provider. |
| `apiKey` | `string` | API key. |
| `model` | `string` | Model override. Defaults: `claude-sonnet-5`, `gpt-4o`, `mistral-large-latest`. |
| `maxTokens` | `number` | Max tokens per LLM call (default: 2048). |
| `adapter` | `(prompt, systemPrompt?) => Promise<string>` | Custom adapter for `"custom"` provider. |

### ToolDefinition

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Unique tool identifier (used by the LLM to call it). |
| `description` | `string` | What the tool does — helps the LLM decide when to use it. |
| `parameters` | `ToolParameter[]` | Parameter schema. |
| `execute` | `(args) => Promise<string>` | Tool implementation. Receives parsed args, returns a string result. |

---

## Supported LLM Providers

| Provider | `provider` value | Default model |
|---|---|---|
| Anthropic | `"anthropic"` | `claude-sonnet-5` |
| OpenAI | `"openai"` | `gpt-4o` |
| Mistral | `"mistral"` | `mistral-large-latest` |
| Any other | `"custom"` | — bring your own adapter |

---

## Hook API

For full control, use `useToolPilot` directly:

```tsx
import { useToolPilot, webSearch, createCodeExecTool } from "tool-pilot";

function Agent() {
  const { run, steps, answer, status, error, reset } = useToolPilot({
    config: {
      llm: { provider: "anthropic", apiKey: "..." },
      tools: [webSearch, createCodeExecTool({ acknowledgeUnsafe: true })],
    },
    onStep: (step) => console.log(`[${step.kind}]`, step.content),
    onComplete: (result) => console.log("Done in", result.totalDurationMs + "ms"),
  });

  return (
    <div>
      <button onClick={() => run("Calculate the 20th Fibonacci number")}>Go</button>

      {steps.map((s) => (
        <div key={s.id}>
          <strong>{s.kind}</strong>: {s.content}
        </div>
      ))}

      {answer && <p><strong>Answer:</strong> {answer}</p>}
      {status === "planning" && <p>Thinking…</p>}
      {status === "executing" && <p>Running tools…</p>}
    </div>
  );
}
```

### Hook Return Value

| Field | Type | Description |
|---|---|---|
| `run` | `(input: string) => Promise<void>` | Start the agent with a goal. |
| `steps` | `ReasoningStep[]` | Live list of reasoning steps (think, tool-call, observation, answer). |
| `answer` | `string` | Final answer once the agent is done. |
| `status` | `AgentStatus` | `"idle"` · `"planning"` · `"executing"` · `"done"` · `"error"` |
| `error` | `Error \| null` | Error, if any. |
| `reset` | `() => void` | Clear all state. |

---

## Headless (No React)

Use `runAgent` directly for Node.js or non-React environments:

```ts
import { runAgent, createCodeExecTool } from "tool-pilot";

const result = await runAgent("What is 2^100?", {
  llm: { provider: "openai", apiKey: "..." },
  tools: [createCodeExecTool({ acknowledgeUnsafe: true })],
  maxSteps: 5,
});

console.log(result.answer);
console.log(`Completed in ${result.steps.length} steps`);
```

---

## Reasoning Step Types

| `kind` | Icon | Description |
|---|---|---|
| `thinking` | 🧠 | LLM reasoning about what to do next |
| `tool-call` | 🔧 | Agent chose a tool and is calling it |
| `observation` | 👁 | Result returned from the tool |
| `answer` | ✅ | Final answer to the user |
| `error` | ❌ | Error during LLM call or tool execution |

---

## License

MIT
