# Contributing to agent-recall

Thanks for your interest! Here's how to get involved.

## Setup

```bash
git clone https://github.com/yourusername/agent-recall
cd agent-recall
pip install -e ".[dev,chroma]"
```

## Running tests

```bash
python tests/test_memory.py
```

No API key needed for the core tests — they only cover the local layers.

## Areas that need help

The roadmap issues are a good starting point. The highest-value contributions:

- **New vector backends** — Qdrant, Weaviate, Redis, Postgres+pgvector
- **Async support** — `AsyncMemory` class wrapping the sync API
- **LangChain wrapper** — drop-in replacement for `ConversationBufferMemory`
- **Importance scoring** — decay/scoring heuristics to auto-prune long-term memory
- **CLI inspector** — `agent-recall inspect <agent_id>` for debugging

## Code style

- `ruff` for linting (`ruff check .`)
- `black` for formatting (`black .`)
- Type hints on all public methods
- Docstrings on all public classes and methods

## PR checklist

- [ ] Tests pass
- [ ] New functionality has tests
- [ ] Docstrings updated
- [ ] README updated if public API changed
