# agentquorum

> Composable multi-agent reasoning pipeline for React. Assemble a council of agents with custom roles, choose sequential or parallel orchestration, plug in any LLM — and get a live visual reasoning trace.

Part of [llm-apps](https://github.com/roaheit/llm-apps) — a growing collection of real-world LLM applications & AI agents.

---

## Installation

```bash
npm install agentquorum
# or
yarn add agentquorum
```

---

## Quick Start

```tsx
import { MultiAgentReasoning } from "agentquorum";

const pipeline = {
  mode: "sequential",
  llm: { provider: "anthropic", apiKey: import.meta.env.VITE_ANTHROPIC_KEY },
  agents: [
    {
      id: "analyst",
      name: "Analyst",
      role: "Break down the problem",
      icon: "📊",
      systemPrompt: "You are a rigorous analyst. Break down the following problem into its core components. Input: {{input}}",
    },
    {
      id: "critic",
      name: "Devil's Advocate",
      role: "Challenge assumptions",
      icon: "⚡",
      systemPrompt: "You are a devil's advocate. The analyst said: {{previous}}. Challenge every assumption and identify risks.",
    },
    {
      id: "researcher",
      name: "Researcher",
      role: "Add evidence and context",
      icon: "🔬",
      systemPrompt: "You are a researcher. Given this context so far:\n{{context}}\n\nAdd supporting evidence, precedents, and relevant context.",
    },
  ],
  synthesizer: {
    name: "Synthesizer",
    icon: "🧠",
    systemPrompt: "You are a master synthesizer. Review all agent outputs below and produce a final, balanced conclusion.\n\n{{outputs}}",
  },
};

export default function App() {
  return (
    <MultiAgentReasoning
      pipeline={pipeline}
      placeholder="What problem should the agents reason about?"
    />
  );
}
```

---

## How It Works

```
User input
    │
    ▼
┌─────────────────────────────────────────┐
│            Pipeline (sequential)         │
│                                         │
│  Agent 1 → Agent 2 → Agent 3 → ...     │
│  (each sees prior context via           │
│   {{previous}} and {{context}})         │
└─────────────────────────────────────────┘
    │
    ▼
Synthesizer (sees all outputs via {{outputs}})
    │
    ▼
Final reasoning output
```

In **parallel** mode, all agents run simultaneously and receive only the original input. The synthesizer still waits for all of them before running.

---

## Pipeline Config

```ts
interface PipelineConfig {
  mode: "sequential" | "parallel";
  agents: AgentConfig[];
  synthesizer: SynthesizerConfig;
  llm: LLMConfig; // fallback for all agents
}
```

### AgentConfig

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier. |
| `name` | `string` | Display name. |
| `role` | `string` | Short description of the agent's purpose. |
| `icon` | `string` | Emoji shown in the timeline. |
| `systemPrompt` | `string` | Prompt defining the agent's behaviour. Supports template variables. |
| `llm` | `LLMConfig` | Optional. Override the pipeline-level LLM for this agent. |

### SynthesizerConfig

| Field | Type | Description |
|---|---|---|
| `name` | `string` | Display name. Default: `"Synthesizer"`. |
| `icon` | `string` | Emoji. Default: `"⚡"`. |
| `systemPrompt` | `string` | Synthesizer prompt. Supports `{{outputs}}` and `{{input}}`. |
| `llm` | `LLMConfig` | Optional. Override the pipeline-level LLM. |

---

## Prompt Template Variables

| Variable | Available in | Description |
|---|---|---|
| `{{input}}` | All agents, synthesizer | The original user input. |
| `{{previous}}` | Sequential agents | The immediately preceding agent's output. |
| `{{context}}` | Sequential agents | All prior agent outputs, formatted. |
| `{{outputs}}` | Synthesizer | All agent outputs, formatted with agent names. |

---

## Supported LLM Providers

| Provider | `provider` value | Default model |
|---|---|---|
| Anthropic | `"anthropic"` | `claude-sonnet-5` |
| OpenAI | `"openai"` | `gpt-4o` |
| Mistral | `"mistral"` | `mistral-large-latest` |
| Any other | `"custom"` | — bring your own adapter |

Each agent can use a **different** provider:

```ts
agents: [
  {
    id: "fast-thinker",
    llm: { provider: "openai", apiKey: "...", model: "gpt-4o-mini" },
    ...
  },
  {
    id: "deep-thinker",
    llm: { provider: "anthropic", apiKey: "...", model: "claude-opus-4-6" },
    ...
  },
]
```

---

## Hook API

For full control, use `useMultiAgent` directly:

```tsx
import { useMultiAgent } from "agentquorum";

const { run, agentResults, synthesis, running, activeAgentId, error, reset } = useMultiAgent({
  pipeline,
  onAgentComplete: (result) => console.log("Agent done:", result.agentName),
  onComplete: (result) => console.log("Pipeline done:", result.totalDurationMs + "ms"),
});

<button onClick={() => run("Should we launch in Q3?")}>Run</button>

{agentResults.map(r => (
  <div key={r.agentId}>
    <strong>{r.agentName}</strong>
    <p>{r.output}</p>
  </div>
))}

{synthesis && <p><strong>Synthesis:</strong> {synthesis}</p>}
```

### Hook Return Value

| Field | Type | Description |
|---|---|---|
| `run` | `(input: string) => Promise<void>` | Start the pipeline. |
| `agentResults` | `AgentResult[]` | Live list of completed agent outputs. |
| `synthesis` | `string` | Final synthesizer output. |
| `running` | `boolean` | True while any agent is active. |
| `activeAgentId` | `string \| null` | ID of the currently running agent. |
| `error` | `Error \| null` | Pipeline-level error, if any. |
| `reset` | `() => void` | Clear all state. |

---

## Component Props

| Prop | Type | Default | Description |
|---|---|---|---|
| `pipeline` | `PipelineConfig` | — | **Required.** The agent pipeline config. |
| `input` | `string` | `""` | Pre-fill the input field. |
| `placeholder` | `string` | built-in | Placeholder for the input textarea. |
| `onComplete` | `(result) => void` | — | Callback when the full pipeline finishes. |
| `onError` | `(error) => void` | — | Callback on pipeline error. |
| `theme` | `"dark" \| "light"` | `"dark"` | Color theme. |
| `className` | `string` | — | Class on the root element. |
| `style` | `CSSProperties` | — | Inline styles on root element. |

---

## API Key Setup

| Use case | Recommendation |
|---|---|
| Local dev / prototyping | `.env` file, never committed |
| Personal or internal tool | `.env` with framework prefix |
| Public-facing production app | Backend proxy — always |

```bash
# .env
VITE_ANTHROPIC_KEY=sk-ant-...
VITE_OPENAI_KEY=sk-...
```

For production, use the `"custom"` provider with an adapter that proxies to your backend.

---

## TypeScript

All types are exported from the package root:

```ts
import type {
  PipelineConfig,
  AgentConfig,
  SynthesizerConfig,
  AgentResult,
  PipelineResult,
  LLMConfig,
} from "agentquorum";
```

---

## License

MIT © [llm-apps](https://github.com/roaheit/llm-apps)
