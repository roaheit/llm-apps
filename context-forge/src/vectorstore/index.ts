import type { DocumentChunk, VectorStoreAdapter, VectorStoreConfig, RetrievedChunk } from "../types";

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * InMemoryVectorStore
 * Pure JS, no dependencies. Uses cosine similarity over Float32 vectors.
 * Not suitable for very large corpora (>100k chunks) — use a real vector DB for that.
 */
export class InMemoryVectorStore implements VectorStoreAdapter {
  private chunks: Map<string, DocumentChunk> = new Map();

  async upsert(chunks: DocumentChunk[]): Promise<void> {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, chunk);
    }
  }

  async query(embedding: number[], topK: number): Promise<DocumentChunk[]> {
    const scored: { chunk: DocumentChunk; score: number }[] = [];
    for (const chunk of this.chunks.values()) {
      if (!chunk.embedding) continue;
      const score = cosineSimilarity(embedding, chunk.embedding);
      scored.push({ chunk, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(({ chunk, score }) => ({
      ...chunk,
      score,
    } as DocumentChunk));
  }

  async delete(documentId: string): Promise<void> {
    for (const [id, chunk] of this.chunks) {
      if (chunk.documentId === documentId) this.chunks.delete(id);
    }
  }

  async clear(): Promise<void> {
    this.chunks.clear();
  }

  async count(): Promise<number> {
    return this.chunks.size;
  }
}

/**
 * Resolves the correct vector store from config.
 * Returns InMemoryVectorStore by default.
 */
export function resolveVectorStore(cfg?: VectorStoreConfig): VectorStoreAdapter {
  if (!cfg || cfg.type === "memory") return new InMemoryVectorStore();
  if (cfg.type === "custom") {
    if (!cfg.adapter) throw new Error("Custom vector store adapter required when type is 'custom'.");
    return cfg.adapter;
  }
  throw new Error(`Unknown vector store type: "${(cfg as VectorStoreConfig).type}"`);
}
