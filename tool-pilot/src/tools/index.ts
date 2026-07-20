import type { ToolDefinition } from "../types";

export { webSearch } from "./webSearch";
export { fileRead } from "./fileRead";
export { createCodeExecTool } from "./codeExec";
export type { CodeExecOptions } from "./codeExec";

export function buildToolDescriptions(tools: ToolDefinition[]): string {
  return tools
    .map((t) => {
      const params = t.parameters
        .map((p) => `    - ${p.name} (${p.type}${p.required ? ", required" : ""}): ${p.description}`)
        .join("\n");
      return `- ${t.name}: ${t.description}\n  Parameters:\n${params}`;
    })
    .join("\n\n");
}
