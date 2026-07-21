import type { AgentResult } from "../types";

export function interpolate(
  template: string,
  vars: {
    input?: string;
    previous?: string;
    context?: string;
    outputs?: string;
  }
): string {
  return template
    .replace(/\{\{input\}\}/g,    vars.input    ?? "")
    .replace(/\{\{previous\}\}/g, vars.previous ?? "")
    .replace(/\{\{context\}\}/g,  vars.context  ?? "")
    .replace(/\{\{outputs\}\}/g,  vars.outputs  ?? "");
}

export function buildContext(results: AgentResult[]): string {
  return results
    .map(r => `[${r.agentName} — ${r.agentRole}]\n${r.output}`)
    .join("\n\n---\n\n");
}

export function buildOutputsSummary(results: AgentResult[]): string {
  return results
    .map(r => `### ${r.icon ?? ""} ${r.agentName} (${r.agentRole})\n\n${r.output}`)
    .join("\n\n---\n\n");
}
