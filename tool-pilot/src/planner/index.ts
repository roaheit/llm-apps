import { callLLM } from "corellm";
import { buildSystemPrompt, parseAgentResponse } from "./prompt";
import type {
  ToolPilotConfig,
  ToolDefinition,
  ReasoningStep,
  AgentRun,
} from "../types";

let stepCounter = 0;
function makeId(): string {
  return `step-${++stepCounter}-${Date.now()}`;
}

export async function runAgent(
  input: string,
  config: ToolPilotConfig,
  onStep?: (step: ReasoningStep) => void
): Promise<AgentRun> {
  const maxSteps = config.maxSteps ?? 10;
  const systemPrompt = buildSystemPrompt(config.tools, config.systemPrompt);

  const steps: ReasoningStep[] = [];
  const toolMap = new Map<string, ToolDefinition>();
  for (const t of config.tools) toolMap.set(t.name, t);

  // Build conversation as a running transcript
  let transcript = `User task: ${input}`;
  const startTime = Date.now();

  for (let i = 0; i < maxSteps; i++) {
    // Ask the LLM to reason
    const stepStart = Date.now();
    let llmOutput: string;
    try {
      llmOutput = await callLLM(transcript, systemPrompt, config.llm);
    } catch (e) {
      const errorStep: ReasoningStep = {
        id: makeId(),
        kind: "error",
        content: `LLM call failed: ${(e as Error).message}`,
        timestamp: Date.now(),
        durationMs: Date.now() - stepStart,
      };
      steps.push(errorStep);
      onStep?.(errorStep);
      return {
        input,
        steps,
        answer: "",
        status: "error",
        totalDurationMs: Date.now() - startTime,
        error: (e as Error).message,
      };
    }

    const parsed = parseAgentResponse(llmOutput);

    // Record thinking step
    if (parsed.think) {
      const thinkStep: ReasoningStep = {
        id: makeId(),
        kind: "thinking",
        content: parsed.think,
        timestamp: Date.now(),
        durationMs: Date.now() - stepStart,
      };
      steps.push(thinkStep);
      onStep?.(thinkStep);
    }

    // If the agent produced a final answer, we're done
    if (parsed.answer) {
      const answerStep: ReasoningStep = {
        id: makeId(),
        kind: "answer",
        content: parsed.answer,
        timestamp: Date.now(),
      };
      steps.push(answerStep);
      onStep?.(answerStep);
      return {
        input,
        steps,
        answer: parsed.answer,
        status: "done",
        totalDurationMs: Date.now() - startTime,
      };
    }

    // If the agent wants to use a tool
    if (parsed.action) {
      const toolCallStep: ReasoningStep = {
        id: makeId(),
        kind: "tool-call",
        content: `Calling ${parsed.action}`,
        toolCall: { tool: parsed.action, args: parsed.actionInput ?? {} },
        timestamp: Date.now(),
      };
      steps.push(toolCallStep);
      onStep?.(toolCallStep);

      const tool = toolMap.get(parsed.action);
      let observation: string;

      if (!tool) {
        observation = `Error: Unknown tool "${parsed.action}". Available tools: ${config.tools.map((t) => t.name).join(", ")}`;
      } else {
        const toolStart = Date.now();
        try {
          observation = await tool.execute(parsed.actionInput ?? {});
        } catch (e) {
          observation = `Tool error: ${(e as Error).message}`;
        }
        toolCallStep.durationMs = Date.now() - toolStart;
      }

      const obsStep: ReasoningStep = {
        id: makeId(),
        kind: "observation",
        content: observation,
        timestamp: Date.now(),
      };
      steps.push(obsStep);
      onStep?.(obsStep);

      // Append the exchange to the transcript so the LLM has context
      transcript += `\n\n${llmOutput}\n\nOBSERVATION: ${observation}`;
      continue;
    }

    // Fallback: LLM didn't follow the format — treat as final answer
    const fallbackStep: ReasoningStep = {
      id: makeId(),
      kind: "answer",
      content: llmOutput.trim(),
      timestamp: Date.now(),
    };
    steps.push(fallbackStep);
    onStep?.(fallbackStep);
    return {
      input,
      steps,
      answer: llmOutput.trim(),
      status: "done",
      totalDurationMs: Date.now() - startTime,
    };
  }

  // Max steps exceeded — force an answer
  const maxStepNote = `(Agent reached the maximum of ${maxSteps} reasoning steps)`;
  const lastOutput = steps.filter((s) => s.kind === "observation" || s.kind === "thinking").at(-1);
  const forcedAnswer = lastOutput
    ? `${maxStepNote}\n\nBased on my research so far:\n${lastOutput.content}`
    : maxStepNote;

  const forcedStep: ReasoningStep = {
    id: makeId(),
    kind: "answer",
    content: forcedAnswer,
    timestamp: Date.now(),
  };
  steps.push(forcedStep);
  onStep?.(forcedStep);

  return {
    input,
    steps,
    answer: forcedAnswer,
    status: "done",
    totalDurationMs: Date.now() - startTime,
  };
}
