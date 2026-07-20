import { extractJson } from "llm-core";
import type { Narration, NarrationRequest } from "./types";

const TONE_GUIDE: Record<string, string> = {
  narrative: "Write flowing, engaging prose, as if telling the story of the data.",
  analyst: "Be precise and businesslike. Lead with findings. Quantify where possible.",
  casual: "Keep it relaxed and friendly, like explaining to a colleague over coffee.",
  teacher: "Explain patiently, defining SQL concepts as they appear, for someone learning SQL.",
};

export function buildPrompt(req: NarrationRequest): string {
  const tone = req.tone ?? "analyst";
  const dialect = req.dialect ?? "ansi";
  const maxRows = req.maxSampleRows ?? 20;

  const parts: string[] = [
    `You are an expert data engineer who explains SQL to humans.`,
    `Dialect: ${dialect}. Tone: ${TONE_GUIDE[tone]}`,
    req.context ? `Business context provided by the user: ${req.context}` : "",
    `SQL query:\n\`\`\`sql\n${req.sql}\n\`\`\``,
  ];

  if (req.results) {
    const { columns, rows, totalRowCount } = req.results;
    const sample = rows.slice(0, maxRows);
    parts.push(
      `Query results (${sample.length} sample row(s)${
        totalRowCount != null ? ` of ${totalRowCount} total` : ""
      }):`,
      `Columns: ${columns.join(", ")}`,
      `Rows:\n${sample.map((r) => JSON.stringify(r)).join("\n")}`
    );
  }

  parts.push(
    `Respond ONLY with a JSON object — no markdown fences, no preamble — with keys:`,
    `"summary" (string, one sentence),`,
    `"queryExplanation" (string, plain-English walkthrough of what the query does),`,
    req.results
      ? `"resultsInterpretation" (string, what the results actually mean),`
      : "",
    `"caveats" (array of strings; empty array if none — e.g. performance smells, ambiguous joins, dialect quirks).`
  );

  return parts.filter(Boolean).join("\n\n");
}

export function parseNarration(raw: string): Narration {
  const data = extractJson<Partial<Narration>>(raw);
  return {
    summary: typeof data.summary === "string" ? data.summary : "",
    queryExplanation: typeof data.queryExplanation === "string" ? data.queryExplanation : "",
    resultsInterpretation:
      typeof data.resultsInterpretation === "string" ? data.resultsInterpretation : undefined,
    caveats: Array.isArray(data.caveats) ? data.caveats : undefined,
  };
}
