import { extractJson } from "llm-core";
import type { ToolDefinition } from "../types";

export function buildSystemPrompt(tools: ToolDefinition[], customPreamble?: string): string {
  const toolBlock = tools
    .map((t) => {
      const params = t.parameters
        .map((p) => `      "${p.name}": <${p.type}>${p.required ? " (required)" : ""} — ${p.description}`)
        .join("\n");
      return `  - ${t.name}: ${t.description}\n    Parameters:\n${params}`;
    })
    .join("\n\n");

  return `${customPreamble ?? "You are a capable AI agent that reasons step-by-step and uses tools to accomplish tasks."}

You have access to the following tools:

${toolBlock}

To use a tool, respond with EXACTLY this format (no extra text around the JSON):

THINK: <your reasoning about what to do next>
ACTION: <tool_name>
ACTION_INPUT: <JSON object with tool parameters>

When you receive the tool result, it will appear as:
OBSERVATION: <tool result>

Continue reasoning and using tools until you have enough information to provide a final answer.
When you are ready to give your final answer, respond with:

THINK: <final reasoning>
ANSWER: <your complete final answer to the user>

Rules:
- Always THINK before taking an action or giving an answer.
- Use one tool at a time. Wait for the observation before continuing.
- If a tool fails, reason about why and try a different approach.
- Do not make up information — use tools to verify facts.
- Keep answers grounded in tool observations.`;
}

export function parseAgentResponse(raw: string): {
  think: string;
  action?: string;
  actionInput?: Record<string, unknown>;
  answer?: string;
} {
  const thinkMatch = raw.match(/THINK:\s*([\s\S]*?)(?=\nACTION:|ANSWER:|$)/i);
  const actionMatch = raw.match(/ACTION:\s*(\S+)/i);
  const answerMatch = raw.match(/ANSWER:\s*([\s\S]*)/i);

  const think = thinkMatch?.[1]?.trim() ?? "";

  if (answerMatch) {
    return { think, answer: answerMatch[1].trim() };
  }

  if (actionMatch) {
    let actionInput: Record<string, unknown> = {};
    const inputIdx = raw.search(/ACTION_INPUT:/i);
    if (inputIdx >= 0) {
      // Balanced JSON scan after ACTION_INPUT: — nested braces no longer truncate the args.
      const after = raw.slice(inputIdx).replace(/ACTION_INPUT:/i, "");
      try {
        actionInput = extractJson<Record<string, unknown>>(after);
      } catch {
        actionInput = {};
      }
    }
    return { think, action: actionMatch[1].trim(), actionInput };
  }

  // If no structured output, treat the whole thing as an answer
  return { think, answer: raw.trim() };
}
