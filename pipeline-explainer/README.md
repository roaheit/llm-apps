# pipeline-explainer

> Paste your Snowflake task DDL (or any DAG as JSON), get an interactive graph **plus** a plain-English narration of how the pipeline flows — powered by **any LLM provider**.

Part of [ilm-apps](https://github.com/ilm-apps) — a collection of useful, opinionated, LLM-agnostic React tools. Siblings: [json-storyteller](https://github.com/ilm-apps/json-storyteller), [sql-narrator](https://github.com/ilm-apps/sql-narrator).

---

## Why

Task graphs are easy to write and hard to read back. `pipeline-explainer` parses `CREATE TASK` DDL — including `AFTER` fan-out/fan-in, `SCHEDULE`, `WHEN SYSTEM$STREAM_HAS_DATA(...)` conditions, and `FINALIZE` tasks — renders the DAG as an SVG, and asks an LLM to narrate the flow in execution order, flag risks, and describe each node.

## Installation

```bash
npm install pipeline-explainer
```

## Quick Start — Snowflake DDL

```tsx
import { PipelineExplainer } from "pipeline-explainer";

<PipelineExplainer
  llm={{ provider: "anthropic", apiKey: import.meta.env.VITE_ANTHROPIC_KEY }}
  context="Nightly load into the sales mart"
  ddl={`
    CREATE OR REPLACE TASK load_root
      WAREHOUSE = etl_wh
      SCHEDULE = 'USING CRON 0 2 * * * UTC'
      AS CALL start_batch();

    CREATE OR REPLACE TASK load_orders
      WAREHOUSE = etl_wh
      AFTER load_root
      WHEN SYSTEM$STREAM_HAS_DATA('orders_stream')
      AS INSERT INTO orders_fact SELECT * FROM orders_stream;

    CREATE OR REPLACE TASK load_customers
      WAREHOUSE = etl_wh
      AFTER load_root
      AS MERGE INTO dim_customer USING stg_customer ON ...;

    CREATE OR REPLACE TASK publish_mart
      WAREHOUSE = etl_wh
      AFTER load_orders, load_customers
      AS CALL refresh_mart();

    CREATE OR REPLACE TASK cleanup
      WAREHOUSE = etl_wh
      FINALIZE = load_root
      AS CALL log_batch_end();
  `}
/>
```

You get: an SVG DAG (roots green, finalizers dashed amber, conditional nodes labeled), click-to-inspect per-node notes, a flow narration, and observations (e.g. "publish_mart fans in from two parents — a failure in either blocks the mart refresh").

## Quick Start — JSON DAG

Any orchestrator's graph works if you can shape it as nodes + dependencies (Airflow, Dagster, dbt, homegrown):

```tsx
<PipelineExplainer
  llm={{ provider: "openai", apiKey: KEY }}
  pipeline={{
    name: "elt_daily",
    nodes: [
      { id: "extract", dependsOn: [], schedule: "0 1 * * *" },
      { id: "transform", dependsOn: ["extract"] },
      { id: "load", dependsOn: ["transform"] },
    ],
  }}
/>
```

## Headless usage

```tsx
import { parseSnowflakeTasks, usePipelineExplainer, DagView } from "pipeline-explainer";

const pipeline = parseSnowflakeTasks(ddlText);
const { narration, explain } = usePipelineExplainer({ provider: "custom", baseUrl: GATEWAY_URL, apiKey: KEY, model: "gpt-5-pro" });
await explain(pipeline, "monthly finance close");
```

## Supported Providers

| Provider | `provider` value | Default model |
|---|---|---|
| Anthropic | `"anthropic"` | `claude-sonnet-5` |
| OpenAI | `"openai"` | `gpt-4o` |
| Mistral | `"mistral"` | `mistral-large-latest` |
| Any other | `"custom"` | — set `baseUrl` to any OpenAI-compatible `/chat/completions` endpoint (LiteLLM, vLLM, Ollama, Azure gateways) |

## Props

| Prop | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLMConfig` | required | Provider, key, model, baseUrl |
| `ddl` | `string` | — | Snowflake `CREATE TASK` script (provide this or `pipeline`) |
| `pipeline` | `Pipeline` | — | `{ nodes: [{ id, dependsOn, schedule?, condition?, finalizer?, body? }] }` |
| `context` | `string` | — | Business context passed to the LLM |
| `auto` | `boolean` | `true` | Explain automatically when input changes |

## Notes

- The DDL parser is pragmatic regex parsing of standard task DDL, not a full SQL parser. For unusual DDL, pre-parse yourself and pass `pipeline`.
- Task bodies are truncated to 500 chars per node before being sent to the LLM.
- **Production apps: use a backend proxy or gateway** — direct browser API calls expose your key. `provider: "custom"` + `baseUrl` points anywhere OpenAI-compatible.

## License

MIT
