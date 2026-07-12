import type { Pipeline } from "./types";

export function buildPipelinePrompt(pipeline: Pipeline, context?: string): string {
  const nodeLines = pipeline.nodes.map((n) =>
    JSON.stringify({
      id: n.id,
      dependsOn: n.dependsOn,
      schedule: n.schedule,
      condition: n.condition,
      warehouse: n.warehouse,
      finalizer: n.finalizer,
      body: n.body ? n.body.slice(0, 500) : undefined,
    })
  );

  return [
    "You are an expert data engineer reviewing a pipeline DAG (e.g. Snowflake task graph).",
    context ? `Context from the user: ${context}` : "",
    `Pipeline nodes (JSON, one per line):\n${nodeLines.join("\n")}`,
    "Explain the pipeline in execution order: which task is the root, where it fans out, where it fans in, what conditions gate execution, and what any finalizer does.",
    'Respond ONLY with a JSON object — no markdown fences, no preamble — with keys:',
    '"summary" (string, one sentence),',
    '"flowExplanation" (string, plain-English walkthrough of the flow in execution order),',
    '"nodeNotes" (object mapping each node id to a one-line description),',
    '"observations" (array of strings: risks or design notes, e.g. missing finalizer, unconditional heavy tasks, fan-in timing hazards; empty array if none).',
  ]
    .filter(Boolean)
    .join("\n\n");
}

export function parsePipelineNarration(raw: string) {
  const clean = raw.replace(/```json|```/g, "").trim();
  return JSON.parse(clean);
}
