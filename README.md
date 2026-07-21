<div align="center">

# 🧠 LLM-Apps

### Production-grade LLM apps, agents & React components — built like software, not demos.

**OpenAI · Anthropic · Mistral · any OpenAI-compatible gateway** — one resilient core, many real tools.

[![CI](https://github.com/roaheit/llm-apps/actions/workflows/ci.yml/badge.svg)](https://github.com/roaheit/llm-apps/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#-license)
[![TypeScript](https://img.shields.io/badge/TypeScript-strict-3178c6.svg)](#)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

**🌐 [Explore the interactive docs →](https://roaheit.github.io/llm-apps)**

</div>

---

> **LLMs are everywhere. Reliable implementations of them are not.**

Getting a model to *say something* is easy. Getting it to do something **useful, repeatable, and safe** — inside a real product — is the hard part. That gap is why this repo exists.

**LLM-Apps** is a growing collection of real-world LLM applications, autonomous agents, and reusable React components — all running on **one shared, resilient engine**, all open source, all held to the standards you'd expect from code you actually ship.

No notebook demos. No copy-paste prompt spaghetti. Just patterns that hold up in production.

---

## ✨ Why it's different

Most "LLM example" repos give you a clever prompt and a `fetch` call. This one is built like a product:

| Most demos | LLM-Apps |
|---|---|
| API call copy-pasted into every file | **One shared `llm-core`** every package speaks through |
| Breaks on a rate limit | **Retries + backoff + timeouts + cancellation** built in |
| Waits, then dumps a wall of text | **Token streaming** where the UX needs it |
| `JSON.parse` and pray | **Resilient structured output** (extract + validate) |
| API keys shipped to the browser | **Proxy-friendly** — keys stay server-side |
| "It works on my machine" | **Tests + CI** on every change |

---

## 🧩 What's inside

### ⚙️ The engine

- **[llm-core](llm-core/)** — the shared, framework-agnostic client every package runs on. One interface for Anthropic, OpenAI, Mistral, or any OpenAI-compatible gateway — with **streaming, retries, timeouts, cancellation, token-usage capture, and structured-output** extraction baked in. Swap providers with one line; keep your keys off the client via `baseUrl`.

### 🤖 Agents & reasoning

- **[tool-pilot](tool-pilot/)** — give an agent a goal and watch it **think → pick a tool → act**, live. A ReAct reasoning loop with pluggable tools (web search, file read, opt-in code exec). · [docs](https://roaheit.github.io/llm-apps/tool-pilot)
- **[agent-council](agent-council/)** — convene a **council of agents** with distinct roles, run them sequentially or in parallel, and watch a live reasoning trace converge on an answer. · [docs](https://roaheit.github.io/llm-apps/agent-council)
- **[agent-recall](agent-recall/)** — give your agents a memory that lasts: **short-term, long-term, and episodic** layers, local-first and framework-agnostic. *(Python)* · [docs](https://roaheit.github.io/llm-apps/agent-recall)

### 📚 Retrieval & data intelligence

- **[context-forge](context-forge/)** — a production-shaped **RAG pipeline in a React hook**: index files, URLs, or text and query with any LLM. In-memory vectors by default; pluggable for Pinecone, Weaviate, pgvector. · [docs](https://roaheit.github.io/llm-apps/context-forge)
- **[sql-narrator](sql-narrator/)** — turn dense SQL — *and what the results actually mean* — into **plain English**. Multi-dialect (Snowflake, Postgres, MySQL, BigQuery…), four tones. · [docs](https://roaheit.github.io/llm-apps/sql-narrator)
- **[pipeline-explainer](pipeline-explainer/)** — paste Snowflake `CREATE TASK` DDL (or any DAG as JSON) and get an **interactive graph + an AI narration** of how it flows: fan-out, fan-in, stream conditions, finalizers, and risks. · [docs](https://roaheit.github.io/llm-apps/pipeline-explainer)

### ✍️ Content

- **[json-storyteller](json-storyteller/)** — turn raw JSON into a **human story** — four tones, any provider, drop-in React component with live token streaming. · [docs](https://roaheit.github.io/llm-apps/json-storyteller)

---

## 🏗️ Built like production

The part most demos skip — and the reason these are reusable:

- 🧱 **One shared core** — no drifting, copy-pasted API calls. Every package talks to LLMs through `llm-core`.
- 🔁 **Resilient by default** — per-request timeouts, exponential backoff on `429`/`5xx`/`529`, and cancellation via `AbortSignal`.
- ⚡ **Streaming** — token-by-token responses where they improve the experience.
- 🧩 **Structured output** — fence-aware, brace-balanced JSON extraction instead of brittle string parsing.
- 🔒 **Security-conscious** — proxy-friendly (keys off the client), **opt-in** code execution, SSRF guards on URL fetches.
- 📊 **Observable** — token usage, finish reason, and resolved model returned on every call.
- ✅ **Tested & CI-gated** — `node:test` suites and GitHub Actions (build · lint · typecheck · test) on every change.
- 📦 **Modern packaging** — dual ESM/CJS builds, first-class types, tree-shakeable, MIT.

---

## 🚀 Getting started

Everything lives in one npm workspace:

```bash
git clone https://github.com/roaheit/llm-apps
cd llm-apps
npm install     # links all packages
npm run build   # build every package
npm test        # run the suites
```

A taste of the shared engine — same call, any provider:

```ts
import { complete, stream } from "llm-core";

const config = { provider: "anthropic", apiKey: process.env.ANTHROPIC_KEY };

// One-shot, with usage + metadata
const { text, usage, model } = await complete(config, {
  prompt: "Explain retrieval-augmented generation in one sentence.",
});

// Or stream it, token by token
await stream(config, {
  prompt: "Write a haiku about databases.",
  onToken: (delta) => process.stdout.write(delta),
});
```

Each package ships its own README and **[interactive docs](https://roaheit.github.io/llm-apps)** — explore the source, compose them, or lift the patterns into your own stack.

> 📦 **Note:** packages aren't on npm yet — use them from source for now. Scoped npm releases are on the roadmap.

---

## 🗺️ Roadmap

**Recently shipped:** shared `llm-core` · token streaming · structured output · security hardening · tests + CI 🎉

| Status | Project | Description |
|---|---|---|
| ✅ Live | **llm-core** | Shared multi-provider LLM client — streaming, retries, structured output, usage |
| ✅ Live | **tool-pilot** | ReAct agent with a live reasoning loop & pluggable tools |
| ✅ Live | **agent-council** | Composable multi-agent reasoning pipeline for React |
| ✅ Live | **agent-recall** | Persistent short/long/episodic memory for LLM agents (Python) |
| ✅ Live | **context-forge** | Retrieval-augmented generation pipeline for React |
| ✅ Live | **sql-narrator** | Explains SQL queries and their results in plain English |
| ✅ Live | **pipeline-explainer** | Visualizes & narrates data pipeline DAGs |
| ✅ Live | **json-storyteller** | LLM-powered JSON → narrative React component |
| 🔜 Next | Scoped npm releases | Publish the packages under a scoped namespace |
| 📋 Planned | Model benchmarking toolkit | Cost-vs-accuracy comparisons across providers & tasks |
| 📋 Planned | Enterprise AI reference architecture | Full-stack blueprint for production AI systems |

---

## 🌐 Docs site

The [`docs/`](docs/) folder is served via **GitHub Pages** at **[roaheit.github.io/llm-apps](https://roaheit.github.io/llm-apps)** — a hub landing page plus interactive docs for each package. To self-host: repo **Settings → Pages → Deploy from branch → `main` → `/docs`**.

---

## 🤝 Contributing

Contributions are welcome — with a quality-first mindset. Keep them **meaningful, maintainable, well-documented, and genuinely useful**. CI (build · lint · typecheck · test) runs on every PR. Gimmicks won't make it in 😊

---

## ⭐ Support

If this is useful to you:

- ⭐ **Star the repo** — it genuinely helps
- 🔁 Share it with someone building AI features
- 🛠️ Build something great with it (and tell me what you made)

---

## 📄 License

[MIT](LICENSE) — free to use, learn from, and build on.

<div align="center">

**AI moves fast. This repo is how I stay ahead — by building, hardening, and sharing in the open.**

*Welcome aboard. Let's build AI that actually ships.*

</div>
