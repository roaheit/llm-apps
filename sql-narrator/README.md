# sql-narrator

> A React component that explains SQL queries — and what their results actually mean — in plain English. Works with **any LLM provider**.

Part of [ilm-apps](https://github.com/ilm-apps) — a collection of useful, opinionated, LLM-agnostic React tools. Sibling of [json-storyteller](https://github.com/ilm-apps/json-storyteller).

---

## Why

You paste a 40-line query into a PR, a runbook, or a Slack thread — and someone always asks "what does this do?" `sql-narrator` answers that question automatically: a one-line summary, a step-by-step walkthrough, an interpretation of the results, and any caveats (ambiguous joins, performance smells, dialect quirks).

## Installation

```bash
npm install sql-narrator
# or
yarn add sql-narrator
```

## Quick Start

```tsx
import { SqlNarrator } from "sql-narrator";

<SqlNarrator
  llm={{ provider: "anthropic", apiKey: import.meta.env.VITE_ANTHROPIC_KEY }}
  dialect="snowflake"
  tone="analyst"
  sql={`
    SELECT region, SUM(amount) AS revenue
    FROM sales
    WHERE order_date >= DATEADD(month, -3, CURRENT_DATE)
    GROUP BY region
    ORDER BY revenue DESC
  `}
  results={{
    columns: ["REGION", "REVENUE"],
    rows: [["EMEA", 1240000], ["AMER", 980000], ["APAC", 445000]],
  }}
/>
```

Or use the hook for full control over rendering:

```tsx
import { useSqlNarrator } from "sql-narrator";

const { narration, loading, error, narrate } = useSqlNarrator({
  provider: "openai",
  apiKey: process.env.NEXT_PUBLIC_OPENAI_KEY,
  model: "gpt-4o",
});

await narrate({ sql: myQuery, dialect: "postgres", tone: "teacher" });
// narration = { summary, queryExplanation, resultsInterpretation?, caveats }
```

## Supported Providers

| Provider | `provider` value | Default model |
|---|---|---|
| Anthropic | `"anthropic"` | `claude-sonnet-5` |
| OpenAI | `"openai"` | `gpt-4o` |
| Mistral | `"mistral"` | `mistral-large-latest` |
| Any other | `"custom"` | — set `baseUrl` to any OpenAI-compatible `/chat/completions` endpoint (LiteLLM, vLLM, Ollama, Azure gateways) |

```tsx
// Example: LiteLLM gateway in front of Azure
<SqlNarrator
  llm={{
    provider: "custom",
    baseUrl: "https://my-gateway.internal/v1/chat/completions",
    apiKey: "sk-litellm-...",
    model: "gpt-5-pro",
  }}
  sql={query}
/>
```

## Props

| Prop | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLMConfig` | required | Provider, key, model, baseUrl |
| `sql` | `string` | required | The query to narrate |
| `results` | `QueryResults` | — | `{ columns, rows, totalRowCount? }` — enables results interpretation |
| `dialect` | `SqlDialect` | `"ansi"` | `snowflake`, `postgres`, `mysql`, `sqlserver`, `bigquery`, `oracle` |
| `tone` | `NarrationTone` | `"analyst"` | `narrative`, `analyst`, `casual`, `teacher` |
| `context` | `string` | — | Business context, e.g. "runs nightly against the sales mart" |
| `maxSampleRows` | `number` | `20` | Cap on result rows sent to the LLM |
| `auto` | `boolean` | `true` | Re-narrate when `sql`/`results` change |

## API Key Setup

Store keys in `.env`, never in source control:

```bash
# Vite
VITE_ANTHROPIC_KEY=sk-ant-...
# Next.js
NEXT_PUBLIC_ANTHROPIC_KEY=sk-ant-...
```

**Production apps: use a backend proxy.** Direct browser API calls expose your key in network requests. Point `provider: "custom"` + `baseUrl` at your own server route or gateway that holds the key server-side.

## License

MIT
