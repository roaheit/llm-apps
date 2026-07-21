# contextsmith

> Production-grade retrieval-augmented generation pipeline for React.
> Index files, URLs, or plain text — query with any LLM. In-memory vector store by default, fully pluggable for Pinecone, Weaviate, pgvector, and more.

Part of [ilm-apps](https://github.com/ilm-apps) — a collection of useful, opinionated React tools.

---

## Installation

```bash
npm install contextsmith
```

---

## Quick Start

```tsx
import { ContextForge } from "contextsmith";

const config = {
  llm: {
    provider: "anthropic",
    apiKey: import.meta.env.VITE_ANTHROPIC_KEY,
  },
  embeddings: {
    provider: "openai",
    apiKey: import.meta.env.VITE_OPENAI_KEY,
  },
};

export default function App() {
  return (
    <ContextForge
      config={config}
      placeholder="Ask anything about your documents…"
    />
  );
}
```

---

## How It Works

```
Documents (file / URL / text)
         │
         ▼
    [ Loader ]         — reads text from any source
         │
         ▼
    [ Chunker ]        — splits into overlapping windows
         │
         ▼
    [ Embedder ]       — embeds each chunk (OpenAI / custom)
         │
         ▼
  [ Vector Store ]     — stores chunks (in-memory or custom)
         │
    Query time:
         │
         ▼
    [ Retriever ]      — embeds query → cosine search → top-k chunks
         │
         ▼
    [ Generator ]      — stuffs context into LLM prompt → answer
         │
         ▼
      Answer + Sources
```

---

## ContextForgeConfig

```ts
const config: ContextForgeConfig = {
  // LLM for answer generation
  llm: {
    provider: "anthropic",          // "anthropic" | "openai" | "mistral" | "custom"
    apiKey: "sk-ant-...",
    model: "claude-sonnet-4-20250514",
    maxTokens: 1500,
    temperature: 0.2,
  },

  // Embedding model
  embeddings: {
    provider: "openai",             // "openai" | "custom"
    apiKey: "sk-...",
    model: "text-embedding-3-small",
  },

  // Vector store — defaults to in-memory
  vectorStore: {
    type: "memory",                 // "memory" | "custom"
  },

  // Chunking strategy
  chunking: {
    chunkSize: 512,                 // characters per chunk
    chunkOverlap: 64,               // overlap between chunks
  },

  // Retrieval config
  retrieval: {
    topK: 5,                        // chunks to retrieve
    scoreThreshold: 0.3,            // minimum cosine similarity
  },

  // Generation config
  generation: {
    systemPrompt: "Answer only from context:\n{{context}}",
    citations: true,
  },
};
```

---

## Document Sources

```ts
import { useRAG } from "contextsmith";

const { index } = useRAG({ config });

// File upload
index([{ type: "file", content: fileObject }]);

// URL
index([{ type: "url", content: "https://example.com/article" }]);

// Raw text
index([{ type: "text", content: "Paste any text here…", metadata: { name: "My Note" } }]);

// Mix and match
index([
  { type: "file",  content: resumeFile },
  { type: "url",   content: "https://docs.example.com/api" },
  { type: "text",  content: internalNote },
]);
```

---

## Plugging in a Custom Vector Store

Use `type: "custom"` to connect Pinecone, Weaviate, pgvector, Qdrant, or any other store:

```ts
import { VectorStoreAdapter } from "contextsmith";

class PineconeAdapter implements VectorStoreAdapter {
  async upsert(chunks) {
    await pinecone.index("my-index").upsert(
      chunks.map(c => ({ id: c.id, values: c.embedding, metadata: { text: c.text } }))
    );
  }

  async query(embedding, topK) {
    const res = await pinecone.index("my-index").query({ vector: embedding, topK });
    return res.matches.map(m => ({ ...m.metadata, id: m.id, score: m.score }));
  }

  async delete(documentId) {
    await pinecone.index("my-index").deleteMany({ documentId });
  }

  async clear() {
    await pinecone.index("my-index").deleteAll();
  }

  async count() {
    const stats = await pinecone.index("my-index").describeIndexStats();
    return stats.totalVectorCount;
  }
}

const config = {
  vectorStore: { type: "custom", adapter: new PineconeAdapter() },
  // ...
};
```

---

## Custom Embedding Adapter

Plug in Cohere, Mistral embeddings, a local model, or your own backend:

```ts
embeddings: {
  provider: "custom",
  adapter: async (text) => {
    const res = await fetch("/api/embed", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    const { embedding } = await res.json();
    return embedding; // number[]
  },
},
```

---

## Custom Chunker

Override the default sliding-window chunker:

```ts
chunking: {
  adapter: (text) => {
    // e.g. split by markdown headings
    return text.split(/^#{1,3}\s/m).filter(Boolean);
  },
},
```

---

## Reranker

Add a reranking step after retrieval (e.g. Cohere Rerank):

```ts
retrieval: {
  topK: 5,
  reranker: async (query, chunks) => {
    const res = await cohere.rerank({ query, documents: chunks.map(c => c.text) });
    return res.results.map(r => chunks[r.index]);
  },
},
```

---

## Hook API

```tsx
import { useRAG } from "contextsmith";

const {
  index,             // (sources: DocumentSource[]) => Promise<void>
  query,             // (question: string) => Promise<void>
  removeDocument,    // (documentId: string) => Promise<void>
  clearIndex,        // () => Promise<void>

  indexedDocuments,  // RawDocument[]
  totalChunks,       // number
  lastResult,        // QueryResult | null
  indexingProgress,  // IndexingProgress
  queryStatus,       // QueryStatus
  error,             // Error | null
} = useRAG({ config });
```

### QueryResult

```ts
interface QueryResult {
  query: string;
  answer: string;
  sources: RetrievedChunk[];   // chunks used for generation, with similarity scores
  durationMs: number;
}
```

---

## Prompt Template Variables

The `generation.systemPrompt` supports two variables:

| Variable | Description |
|---|---|
| `{{context}}` | Retrieved chunks, formatted with source names |
| `{{query}}` | The user's question |

---

## API Key Setup

| Use case | Recommendation |
|---|---|
| Local dev / prototyping | `.env` file, never committed |
| Internal tool | `.env` with framework prefix |
| Production app | Backend proxy — keep keys server-side |

```bash
# .env
VITE_ANTHROPIC_KEY=sk-ant-...
VITE_OPENAI_KEY=sk-...
```

For production, use `provider: "custom"` adapters that proxy to your own backend.

---

## TypeScript

All types exported from the package root:

```ts
import type {
  ContextForgeConfig, DocumentSource, QueryResult,
  VectorStoreAdapter, EmbeddingConfig, LLMConfig,
} from "contextsmith";
```

---

## License

MIT © [ilm-apps](https://github.com/ilm-apps)
