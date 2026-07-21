import type {
  EmbeddingConfig,
  VectorStoreAdapter,
  RetrievalConfig,
  RetrievedChunk,
} from "../types";
import { embed } from "../embeddings";

export async function retrieve(
  query: string,
  embeddingCfg: EmbeddingConfig,
  store: VectorStoreAdapter,
  cfg: RetrievalConfig = {}
): Promise<RetrievedChunk[]> {
  const topK      = cfg.topK ?? 5;
  const threshold = cfg.scoreThreshold ?? 0.0;

  // Embed the query
  const queryEmbedding = await embed(query, embeddingCfg);

  // Vector search
  const raw = await store.query(queryEmbedding, topK * 2); // over-fetch for reranking

  // Apply score threshold
  const filtered = (raw as RetrievedChunk[]).filter(c => (c.score ?? 0) >= threshold);

  // Optional reranker (e.g. cross-encoder, Cohere rerank)
  const reranked = cfg.reranker
    ? await cfg.reranker(query, filtered)
    : filtered;

  return (reranked as RetrievedChunk[]).slice(0, topK);
}
