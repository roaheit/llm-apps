
# 🧠 LLM-Apps

### A Curated Collection of Real-World LLM Applications & Autonomous AI Agents

**🌐 [llm-apps.github.io](https://roaheit.github.io/llm-apps)** — browse the interactive docs site

Modern Large Language Models are powerful — but the real challenge is turning that power into **usable, reliable, production-ready applications**.
This repository exists to bridge that gap.

**LLM-Apps** is a growing collection of practical AI applications, intelligent agents, reusable components, and architectural patterns built using **OpenAI, Anthropic, Google Gemini, and leading open-source models**.

---

## 🚀 What This Repository Is

A hands-on playground and learning hub to:

* Build **real AI products**, not toy demos
* Explore **multi-model strategies**
* Design **LLM Agents with reasoning, tools, memory & planning**
* Learn **clean architecture for AI systems**
* Experiment, iterate, and scale powerful AI capabilities

This repo will evolve continuously — driven by real problems, real experimentation, and lessons learned.

---

## ❓ Why This Exists

LLMs are everywhere. **Good implementations are not.**

Most teams struggle with:

* messy, unstructured examples
* copy-paste “prompt engineering”
* unclear architecture & best practices
* difficulty moving from POC → Production
* confusion on choosing the right model or approach

This repository solves that by focusing on:

* clean, modular, understandable code
* opinionated architecture that actually works
* clearly documented reasoning
* real-world, meaningful use cases

---

## 🧩 What You’ll Find Here

(Examples — will expand over time)

### 🔹 Agents

* Task-planning autonomous agents
* Multi-tool reasoning agents
* Retrieval-augmented agents (RAG)
* Workflow & business automation agents
* Data analysis / SQL intelligence agents

### 🔹 Apps

* Productivity assistants
* Knowledge management tools
* Business & enterprise utilities
* Developer tools
* Domain-specific AI applications

### 🔹 Foundations

* Architecture blueprints
* Prompt patterns & libraries
* Evaluation & testing approaches
* Cost vs accuracy strategies
* Deployment & performance guidance

### 🔹 Libraries & Components

* **agent-recall** — Drop-in persistent memory for LLM agents. Short-term, long-term, and episodic memory layers with local-first, framework-agnostic design.
  * Source: [agent-recall/](agent-recall/)
  * Docs: [agent-recall/README.md](agent-recall/README.md) · [interactive docs](https://roaheit.github.io/llm-apps/agent-recall)

* **json-storyteller** — React component that transforms raw JSON into human-readable narratives using any LLM provider (Anthropic, OpenAI, Mistral, custom).
  * Source: [json-storyteller/](json-storyteller/)
  * Docs: [json-storyteller/README.md](json-storyteller/README.md) · [interactive docs](https://roaheit.github.io/llm-apps/json-storyteller)

* **agent-council** — Composable multi-agent reasoning pipeline for React. Assemble a council of agents with custom roles, choose sequential or parallel orchestration, and get a live visual reasoning trace.
  * Source: [agent-council/](agent-council/)
  * Docs: [agent-council/README.md](agent-council/README.md) · [interactive docs](https://roaheit.github.io/llm-apps/agent-council)

* **context-forge** — Production-grade retrieval-augmented generation pipeline for React. Index files, URLs, or plain text — query with any LLM. In-memory vector store by default, fully pluggable for Pinecone, Weaviate, pgvector, and more.
  * Source: [context-forge/](context-forge/)
  * Docs: [context-forge/README.md](context-forge/README.md) · [interactive docs](https://roaheit.github.io/llm-apps/context-forge)

---

## 🏗️ Tech Philosophy

Use the right tool for the right job.

Expect to see:

* **OpenAI / Anthropic / Gemini**
* **Open-source models where they make sense**
* **RAG with vector databases**
* **Backend-first clean architectures**
* **Reusable utilities instead of hacks**

Priorities:
✔ Reliability
✔ Practicality
✔ Scalability
✔ Real-world usability

---

## 🗺️ Roadmap

| Status | Project | Description |
|---|---|---|
| ✅ Live | **agent-recall** | Persistent memory library for LLM agents |
| ✅ Live | **json-storyteller** | React component for LLM-powered JSON narration |
| ✅ Live | **agent-council** | Composable multi-agent reasoning pipeline for React |
| 🔜 Soon | Multi-tool reasoning agent | Task-planning agent with web search, code exec & file tools |
| 🔜 Soon | RAG reference implementation | Production-grade retrieval-augmented generation pipeline |
| 📋 Planned | Model benchmarking toolkit | Cost vs accuracy comparisons across providers & tasks |
| 📋 Planned | Enterprise AI reference architecture | Full-stack blueprint for production AI systems |
| 📋 Planned | Deployment & performance guide | Practical patterns for shipping LLM apps to production |

---

## 🌐 GitHub Pages (Docs Site)

The [`docs/`](docs/) folder is served as a GitHub Pages site. Enable it in your repo settings under **Pages → Source → Deploy from branch → `main` → `/docs`**.

```
docs/
├── index.html              ← hub landing page (all projects)
├── agent-recall/
│   └── index.html          ← agent-recall interactive docs
├── json-storyteller/
│   └── index.html          ← json-storyteller interactive docs
└── agent-council/
    └── index.html          ← agent-council interactive docs
```

---

## 🤝 Contributing

Contributions are welcome — with a quality-first mindset.

Please ensure contributions are:

* meaningful
* maintainable
* well-documented
* actually useful

Gimmicks won’t make it in 😊

---

## ⭐ Support

If you find value here:

* ⭐ Star the repo
* Share it
* Build something awesome with it

---

## 📌 Final Thought

AI is evolving fast. This repo is my way of staying ahead — by **building, experimenting, and sharing**, not just talking.

Welcome aboard. Let’s build real AI.
