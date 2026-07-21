import type { RetrievedChunk, GenerationConfig, LLMConfig, QueryResult } from "../types";
import { stream } from "corellm";

const DEFAULT_SYSTEM = `You are a precise, helpful assistant that answers questions strictly based on the provided context.

Rules:
- Only use information from the provided context to answer
- If the context doesn't contain enough information, say so clearly
- Be concise and direct
- When citing sources, reference the document name

Context:
{{context}}`;

function buildContext(chunks: RetrievedChunk[]): string {
  return chunks
    .map((c, i) =>
      `[Source ${i + 1}: ${c.sourceName}]\n${c.text}`
    )
    .join("\n\n---\n\n");
}

export async function generate(
  query: string,
  chunks: RetrievedChunk[],
  llmCfg: LLMConfig,
  genCfg: GenerationConfig = {},
  startTime: number,
  onToken?: (delta: string, accumulated: string) => void
): Promise<QueryResult> {
  const context = buildContext(chunks);
  const systemTemplate = genCfg.systemPrompt ?? DEFAULT_SYSTEM;
  const system = systemTemplate
    .replace(/\{\{context\}\}/g, context)
    .replace(/\{\{query\}\}/g, query);

  const userPrompt = genCfg.citations !== false
    ? `${query}\n\nPlease cite the source names when referencing specific information.`
    : query;

  const { text: answer } = await stream(llmCfg, { prompt: userPrompt, system, onToken });

  return {
    query,
    answer,
    sources: chunks,
    durationMs: Date.now() - startTime,
  };
}
