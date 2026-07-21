import { useState, useCallback, useRef } from "react";
import { loadSource } from "../pipeline/loader";
import { chunkDocument } from "../chunker";
import { embedBatch } from "../embeddings";
import { resolveVectorStore } from "../vectorstore";
import { retrieve } from "../retriever";
import { generate } from "../pipeline/generator";
import type {
  UseRAGOptions,
  UseRAGReturn,
  DocumentSource,
  RawDocument,
  DocumentChunk,
  IndexingProgress,
  QueryStatus,
  VectorStoreAdapter,
  QueryResult,
} from "../types";

export function useRAG({
  config,
  onIndexed,
  onQueried,
  onError,
}: UseRAGOptions): UseRAGReturn {
  const [indexedDocuments, setIndexedDocuments] = useState<RawDocument[]>([]);
  const [totalChunks, setTotalChunks]           = useState(0);
  const [lastResult, setLastResult]             = useState<QueryResult | null>(null);
  const [indexingProgress, setIndexingProgress] = useState<IndexingProgress>({ status: "idle" });
  const [queryStatus, setQueryStatus]           = useState<QueryStatus>("idle");
  const [error, setError]                       = useState<Error | null>(null);

  // Stable store reference across renders
  const storeRef = useRef<VectorStoreAdapter>(
    resolveVectorStore(config.vectorStore)
  );

  // ── INDEX ──────────────────────────────────────────────────────────────────
  const index = useCallback(async (sources: DocumentSource[]) => {
    setError(null);

    for (const source of sources) {
      try {
        // 1. Load
        setIndexingProgress({ status: "loading", documentName: typeof source.content === "string" ? source.content.slice(0, 60) : (source.content as File).name });
        const doc = await loadSource(source);

        // 2. Chunk
        setIndexingProgress({ status: "chunking", documentName: doc.sourceName });
        const chunks: DocumentChunk[] = chunkDocument(doc, config.chunking);

        // 3. Embed
        setIndexingProgress({ status: "embedding", documentName: doc.sourceName, chunksProcessed: 0, totalChunks: chunks.length });
        const texts = chunks.map(c => c.text);
        const embeddings = await embedBatch(
          texts,
          config.embeddings,
          (done, total) => setIndexingProgress(p => ({ ...p, chunksProcessed: done, totalChunks: total }))
        );

        const chunksWithEmbeddings: DocumentChunk[] = chunks.map((c, i) => ({
          ...c,
          embedding: embeddings[i],
        }));

        // 4. Store
        setIndexingProgress({ status: "storing", documentName: doc.sourceName });
        await storeRef.current.upsert(chunksWithEmbeddings);

        // 5. Update state
        setIndexedDocuments(prev => [...prev.filter(d => d.id !== doc.id), doc]);
        setTotalChunks(await storeRef.current.count());
        setIndexingProgress({ status: "done", documentName: doc.sourceName, chunksProcessed: chunks.length, totalChunks: chunks.length });

        onIndexed?.(doc, chunksWithEmbeddings);
      } catch (e) {
        const err = e instanceof Error ? e : new Error(String(e));
        setIndexingProgress({ status: "error", error: err.message });
        setError(err);
        onError?.(err);
      }
    }
  }, [config, onIndexed, onError]);

  // ── QUERY ──────────────────────────────────────────────────────────────────
  const query = useCallback(async (question: string) => {
    setError(null);
    setQueryStatus("embedding");
    const startTime = Date.now();

    try {
      // 1. Retrieve
      setQueryStatus("retrieving");
      const retrieved = await retrieve(
        question,
        config.embeddings,
        storeRef.current,
        config.retrieval
      );

      // 2. Generate
      setQueryStatus("generating");
      const result = await generate(
        question,
        retrieved,
        config.llm,
        config.generation,
        startTime,
        (_delta, acc) =>
          setLastResult({ query: question, answer: acc, sources: retrieved, durationMs: Date.now() - startTime })
      );

      setLastResult(result);
      setQueryStatus("done");
      onQueried?.(result);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      setQueryStatus("error");
      setError(err);
      onError?.(err);
    }
  }, [config, onQueried, onError]);

  // ── REMOVE / CLEAR ─────────────────────────────────────────────────────────
  const removeDocument = useCallback(async (documentId: string) => {
    await storeRef.current.delete(documentId);
    setIndexedDocuments(prev => prev.filter(d => d.id !== documentId));
    setTotalChunks(await storeRef.current.count());
  }, []);

  const clearIndex = useCallback(async () => {
    await storeRef.current.clear();
    setIndexedDocuments([]);
    setTotalChunks(0);
    setLastResult(null);
    setIndexingProgress({ status: "idle" });
    setQueryStatus("idle");
  }, []);

  return {
    index,
    query,
    removeDocument,
    clearIndex,
    indexedDocuments,
    totalChunks,
    lastResult,
    indexingProgress,
    queryStatus,
    error,
  };
}
