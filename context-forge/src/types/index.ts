// ─── LLM ────────────────────────────────────────────────────────────────────

export type LLMProvider = "anthropic" | "openai" | "mistral" | "custom";

export interface LLMConfig {
  provider: LLMProvider;
  apiKey: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
  adapter?: (prompt: string, systemPrompt?: string) => Promise<string>;
}

// ─── Embeddings ──────────────────────────────────────────────────────────────

export type EmbeddingProvider = "openai" | "custom";

export interface EmbeddingConfig {
  provider: EmbeddingProvider;
  apiKey?: string;
  model?: string;
  /** Custom embedding function — receives text, returns float vector */
  adapter?: (text: string) => Promise<number[]>;
}

// ─── Documents & Chunks ──────────────────────────────────────────────────────

export type DocumentSourceType = "file" | "url" | "text";

export interface DocumentSource {
  type: DocumentSourceType;
  /** File object (for type "file"), URL string (for "url"), raw text (for "text") */
  content: File | string;
  /** Optional metadata attached to every chunk from this source */
  metadata?: Record<string, unknown>;
}

export interface RawDocument {
  id: string;
  sourceType: DocumentSourceType;
  sourceName: string;
  rawText: string;
  metadata?: Record<string, unknown>;
}

export interface DocumentChunk {
  id: string;
  documentId: string;
  sourceName: string;
  text: string;
  embedding?: number[];
  /** Character offset in the original document */
  startChar: number;
  endChar: number;
  chunkIndex: number;
  metadata?: Record<string, unknown>;
}

// ─── Chunking ────────────────────────────────────────────────────────────────

export interface ChunkingConfig {
  /** Chunk size in characters. @default 512 */
  chunkSize?: number;
  /** Overlap between chunks in characters. @default 64 */
  chunkOverlap?: number;
  /** Custom chunker — receives raw text, returns array of text chunks */
  adapter?: (text: string) => string[];
}

// ─── Vector Store ────────────────────────────────────────────────────────────

export interface VectorStoreAdapter {
  /** Add chunks (with embeddings) to the store */
  upsert(chunks: DocumentChunk[]): Promise<void>;
  /** Query the store — returns top-k most similar chunks */
  query(embedding: number[], topK: number): Promise<DocumentChunk[]>;
  /** Remove all chunks for a given documentId */
  delete(documentId: string): Promise<void>;
  /** Clear the entire store */
  clear(): Promise<void>;
  /** Return total chunk count */
  count(): Promise<number>;
}

export interface VectorStoreConfig {
  /**
   * "memory" uses the built-in in-memory cosine similarity store.
   * "custom" lets you plug in Pinecone, Weaviate, pgvector, etc.
   */
  type: "memory" | "custom";
  adapter?: VectorStoreAdapter;
}

// ─── Retrieval ───────────────────────────────────────────────────────────────

export interface RetrievalConfig {
  /** Number of chunks to retrieve. @default 5 */
  topK?: number;
  /** Minimum cosine similarity score (0–1). @default 0.0 */
  scoreThreshold?: number;
  /** Reranker function — optionally rerank retrieved chunks before generation */
  reranker?: (query: string, chunks: DocumentChunk[]) => Promise<DocumentChunk[]>;
}

export interface RetrievedChunk extends DocumentChunk {
  score: number;
}

// ─── Generation ──────────────────────────────────────────────────────────────

export interface GenerationConfig {
  /**
   * System prompt template for the RAG generator.
   * Use {{context}} for retrieved chunks and {{query}} for user query.
   */
  systemPrompt?: string;
  /** Whether to include source citations in the response. @default true */
  citations?: boolean;
}

// ─── Pipeline ────────────────────────────────────────────────────────────────

export interface ContextForgeConfig {
  llm: LLMConfig;
  embeddings: EmbeddingConfig;
  vectorStore?: VectorStoreConfig;
  chunking?: ChunkingConfig;
  retrieval?: RetrievalConfig;
  generation?: GenerationConfig;
}

// ─── Runtime state ───────────────────────────────────────────────────────────

export type IndexingStatus = "idle" | "loading" | "chunking" | "embedding" | "storing" | "done" | "error";
export type QueryStatus    = "idle" | "embedding" | "retrieving" | "generating" | "done" | "error";

export interface IndexingProgress {
  status: IndexingStatus;
  documentName?: string;
  chunksProcessed?: number;
  totalChunks?: number;
  error?: string;
}

export interface QueryResult {
  query: string;
  answer: string;
  sources: RetrievedChunk[];
  durationMs: number;
}

// ─── Hook ────────────────────────────────────────────────────────────────────

export interface UseRAGOptions {
  config: ContextForgeConfig;
  onIndexed?: (doc: RawDocument, chunks: DocumentChunk[]) => void;
  onQueried?: (result: QueryResult) => void;
  onError?: (error: Error) => void;
}

export interface UseRAGReturn {
  /** Index one or more document sources */
  index: (sources: DocumentSource[]) => Promise<void>;
  /** Query the indexed documents */
  query: (question: string) => Promise<void>;
  /** Remove a document by id */
  removeDocument: (documentId: string) => Promise<void>;
  /** Clear all indexed documents */
  clearIndex: () => Promise<void>;

  indexedDocuments: RawDocument[];
  totalChunks: number;
  lastResult: QueryResult | null;
  indexingProgress: IndexingProgress;
  queryStatus: QueryStatus;
  error: Error | null;
}

// ─── Component ───────────────────────────────────────────────────────────────

export interface ContextForgeProps {
  config: ContextForgeConfig;
  onQueried?: (result: QueryResult) => void;
  onError?: (error: Error) => void;
  theme?: "dark" | "light";
  className?: string;
  style?: React.CSSProperties;
  /** Initial placeholder question */
  placeholder?: string;
}
