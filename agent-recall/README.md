# 🧠 agent-recall

**Drop-in persistent memory for LLM agents.**  
Short-term, long-term, and episodic layers — framework-agnostic, local-first, no cloud required.

```python
from agent_recall import Memory

mem = Memory(agent_id="my-agent")
mem.remember("User prefers concise bullet-point answers", layer="long_term")
context = mem.recall("What are the user's preferences?")
mem.log_episode(action="web_search", result="Found 3 papers", tags=["research"])
```

---

## Why this exists

Every agent framework reinvents memory poorly. LangChain's memory is tightly coupled to its chain abstraction. Most tutorials just stuff the full conversation into the context window and hope for the best.

`agent-recall` is a **standalone, framework-agnostic** memory library that handles all three memory types properly:

| Layer | What it stores | Where | Lifespan |
|---|---|---|---|
| **Short-term** | Recent messages, current session context | In-process | Session |
| **Long-term** | Facts, preferences, conclusions | Local vector DB | Persistent |
| **Episodic** | Timestamped action/outcome log | SQLite / JSONL | Persistent |

---

## Install

```bash
pip install agent-recall

# With ChromaDB for local semantic search (recommended)
pip install "agent-recall[chroma]"

# With Pinecone
pip install "agent-recall[pinecone]"
```

---

## Quickstart

```python
import os
from agent_recall import Memory

mem = Memory(agent_id="research-agent")  # memories persist to .agent_recall/

# Remember something long-term
mem.remember("User's name is Alex and they prefer Python", layer="long_term")

# Retrieve relevant context before an LLM call
context = mem.recall("What do I know about the user?")
# → "[Relevant long-term memories]\n- User's name is Alex and they prefer Python"

# Use context in your LLM call
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": f"Context:\n{context}"},
        {"role": "user", "content": "Remind me of my preferences."},
    ],
)

# Log agent actions
mem.log_episode(
    action="database_query",
    result="Retrieved 42 records",
    tags=["data", "sql"],
)

# When the short-term buffer fills up, compress it
if mem.short_term.is_full():
    mem.compress()  # summarizes → long-term, clears buffer

# Check memory state
print(mem.stats())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Memory(agent_id)                  │
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Short-term  │  │  Long-term   │  │ Episodic  │  │
│  │   (buffer)   │  │  (semantic)  │  │  (log)    │  │
│  │              │  │              │  │           │  │
│  │ In-process   │  │ ChromaDB     │  │ JSONL     │  │
│  │ list         │  │ (local)      │  │ SQLite    │  │
│  │              │  │              │  │           │  │
│  │ auto-compress│  │ vector search│  │ filterable│  │
│  │ when full    │  │ cross-session│  │ by tag/   │  │
│  └──────┬───────┘  └──────────────┘  │ time/     │  │
│         │                            │ action    │  │
│         └── compress() ──────────────┘           │  │
│              (LLM summary → long-term)            │  │
└─────────────────────────────────────────────────────┘
```

---

## Core API

### `Memory(agent_id, **kwargs)`

| Param | Default | Description |
|---|---|---|
| `agent_id` | required | Unique ID for this agent's memory namespace |
| `storage_dir` | `.agent_recall` | Root directory for persistent storage |
| `embedding_model` | `text-embedding-3-small` | OpenAI embedding model |
| `llm_model` | `gpt-4o-mini` | Model used for compression |
| `max_short_term_tokens` | `2000` | Buffer size before `is_full()` returns True |
| `top_k_recall` | `5` | Number of long-term results returned per search |

### Methods

```python
mem.remember(content, layer="long_term", metadata={})  # store
mem.recall(query, layers=["short_term", "long_term"])  # retrieve
mem.log_episode(action, result, tags=[], metadata={})  # log action
mem.compress()                                          # summarize + clear buffer
mem.clear(layer=None)                                   # wipe memory
mem.stats()                                             # inspection
```

---

## Swapping the vector backend

Implement the base interface and pass it in:

```python
from agent_recall.backends.base import VectorBackend

class MyPineconeBackend(VectorBackend):
    def add(self, content, embedding, metadata): ...
    def search(self, query_embedding, top_k): ...
    def clear(self): ...
    def count(self): ...

mem = Memory(agent_id="my-agent", vector_backend=MyPineconeBackend(...))
```

Built-in backends: `ChromaBackend` (default), `PineconeBackend`, `QdrantBackend`.

---

## Comparison

| | agent-recall | LangChain Memory | MemGPT |
|---|---|---|---|
| Framework-agnostic | ✅ | ❌ | ❌ |
| 3 memory layers | ✅ | ❌ | Partial |
| Local-first | ✅ | ✅ | ❌ |
| Swappable backends | ✅ | Partial | ❌ |
| pip install + go | ✅ | ✅ | ❌ |
| Episodic log | ✅ | ❌ | ❌ |

---

## Examples

- [`simple_chatbot.py`](examples/simple_chatbot.py) — chatbot that remembers facts across sessions
- [`research_agent.py`](examples/research_agent.py) — multi-session research agent (the killer demo)

---

## Roadmap

- [ ] Redis backend for short-term layer
- [ ] Memory TTL / expiration policies  
- [ ] Importance scoring (auto-forget low-signal memories)
- [ ] LangChain integration wrapper
- [ ] Async support (`AsyncMemory`)
- [ ] Memory visualization / inspector CLI

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
