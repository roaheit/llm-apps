import type { RawDocument, DocumentChunk, ChunkingConfig } from "../types";

let chunkCounter = 0;

function makeChunkId(docId: string, index: number) {
  return `${docId}_chunk_${index}_${++chunkCounter}`;
}

/**
 * Splits text into overlapping character-level windows.
 * Respects sentence boundaries where possible.
 */
function slidingWindow(
  text: string,
  chunkSize: number,
  overlap: number
): { text: string; start: number; end: number }[] {
  const results: { text: string; start: number; end: number }[] = [];
  let start = 0;

  while (start < text.length) {
    let end = Math.min(start + chunkSize, text.length);

    // Try to snap to a sentence boundary (. ! ?) to avoid cutting mid-sentence
    if (end < text.length) {
      const snapped = text.lastIndexOf(".", end);
      if (snapped > start + chunkSize / 2) end = snapped + 1;
    }

    const chunk = text.slice(start, end).trim();
    if (chunk.length > 0) {
      results.push({ text: chunk, start, end });
    }

    if (end >= text.length) break;
    start = end - overlap;
  }

  return results;
}

export function chunkDocument(
  doc: RawDocument,
  cfg: ChunkingConfig = {}
): DocumentChunk[] {
  const chunkSize = cfg.chunkSize ?? 512;
  const overlap   = cfg.chunkOverlap ?? 64;

  const windows = cfg.adapter
    ? cfg.adapter(doc.rawText).map((text, i) => ({
        text,
        start: 0,
        end: text.length,
      }))
    : slidingWindow(doc.rawText, chunkSize, overlap);

  return windows.map((w, i) => ({
    id:          makeChunkId(doc.id, i),
    documentId:  doc.id,
    sourceName:  doc.sourceName,
    text:        w.text,
    startChar:   w.start,
    endChar:     w.end,
    chunkIndex:  i,
    metadata:    doc.metadata,
  }));
}
