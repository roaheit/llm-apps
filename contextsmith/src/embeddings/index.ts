import type { EmbeddingConfig } from "../types";

async function openaiEmbed(text: string, cfg: EmbeddingConfig): Promise<number[]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify({
      model: cfg.model ?? "text-embedding-3-small",
      input: text,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message ?? `OpenAI embeddings ${res.status}`);
  return data.data?.[0]?.embedding ?? [];
}

/**
 * Lightweight fallback TF-IDF-style hash embedding.
 * Deterministic, no API required — useful for testing or offline mode.
 * Not semantically meaningful for production use.
 */
export function hashEmbed(text: string, dims = 384): number[] {
  const vec = new Float32Array(dims).fill(0);
  const words = text.toLowerCase().split(/\W+/).filter(Boolean);
  for (const word of words) {
    let h = 5381;
    for (let i = 0; i < word.length; i++) h = ((h << 5) + h) ^ word.charCodeAt(i);
    const idx = Math.abs(h) % dims;
    vec[idx] += 1;
  }
  // L2 normalise
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1;
  return Array.from(vec.map(v => v / norm));
}

export async function embed(text: string, cfg: EmbeddingConfig): Promise<number[]> {
  if (cfg.provider === "custom") {
    if (!cfg.adapter) throw new Error("Custom embedding adapter required.");
    return cfg.adapter(text);
  }
  if (cfg.provider === "openai") return openaiEmbed(text, cfg);
  throw new Error(`Unknown embedding provider: "${cfg.provider}"`);
}

export async function embedBatch(
  texts: string[],
  cfg: EmbeddingConfig,
  onProgress?: (done: number, total: number) => void
): Promise<number[][]> {
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i++) {
    results.push(await embed(texts[i], cfg));
    onProgress?.(i + 1, texts.length);
  }
  return results;
}
