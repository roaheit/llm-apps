"""Long-term (semantic / vector) memory layer."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


class LongTermMemory:
    """
    Cross-session semantic memory using embeddings + a local vector store.

    Default backend: ChromaDB (local, no server needed).
    Swap via the backends module for Pinecone, Qdrant, etc.
    """

    def __init__(
        self,
        agent_id: str,
        storage_dir: str,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        top_k: int = 5,
    ):
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.top_k = top_k
        self._collection = None
        self._client = None
        self._init_store()

    def _init_store(self) -> None:
        """Lazily initialise ChromaDB. Falls back to a JSON flat-file store if Chroma isn't installed."""
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.storage_dir)
            self._collection = self._client.get_or_create_collection(
                name=f"agent_recall_{self.agent_id}",
                metadata={"hnsw:space": "cosine"},
            )
            self._backend = "chroma"
        except ImportError:
            # Fallback: plain JSON store (no semantic search, just keyword match)
            self._backend = "json"
            self._json_path = os.path.join(self.storage_dir, "long_term.json")
            if not os.path.exists(self._json_path):
                with open(self._json_path, "w") as f:
                    json.dump([], f)

    def _embed(self, text: str) -> list[float]:
        """Get an embedding vector for text via OpenAI."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def add(self, content: str, metadata: Optional[dict] = None) -> None:
        """Embed and store a memory."""
        meta = metadata or {}
        meta["timestamp"] = time.time()

        if self._backend == "chroma":
            doc_id = f"{self.agent_id}_{int(meta['timestamp'] * 1000)}"
            embedding = self._embed(content)
            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta],
            )
        else:
            # JSON fallback
            with open(self._json_path) as f:
                store = json.load(f)
            store.append({"content": content, "metadata": meta})
            with open(self._json_path, "w") as f:
                json.dump(store, f, indent=2)

    def search(self, query: str) -> list[str]:
        """Return top-k semantically relevant memories for the query."""
        if self._backend == "chroma":
            if self._collection.count() == 0:
                return []
            query_embedding = self._embed(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.top_k, self._collection.count()),
            )
            return results["documents"][0] if results["documents"] else []
        else:
            # JSON fallback: simple keyword overlap
            with open(self._json_path) as f:
                store = json.load(f)
            query_words = set(query.lower().split())
            scored = []
            for entry in store:
                content_words = set(entry["content"].lower().split())
                score = len(query_words & content_words)
                if score > 0:
                    scored.append((score, entry["content"]))
            scored.sort(reverse=True)
            return [content for _, content in scored[: self.top_k]]

    def clear(self) -> None:
        """Delete all memories from the store."""
        if self._backend == "chroma":
            self._client.delete_collection(f"agent_recall_{self.agent_id}")
            self._collection = self._client.get_or_create_collection(
                name=f"agent_recall_{self.agent_id}",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            with open(self._json_path, "w") as f:
                json.dump([], f)

    def stats(self) -> dict[str, Any]:
        if self._backend == "chroma":
            return {"backend": "chroma", "count": self._collection.count()}
        else:
            with open(self._json_path) as f:
                store = json.load(f)
            return {"backend": "json_fallback", "count": len(store)}
