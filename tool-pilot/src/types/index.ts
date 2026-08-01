import type { LLMConfig, LLMProvider, TokenUsage } from "corellm";
export type { LLMConfig, LLMProvider, TokenUsage };
import type { LoopGuardConfig, HaltReason } from "loopguard";
export type { LoopGuardConfig, HaltReason };

export type StepKind = "thinking" | "tool-call" | "observation" | "answer" | "error" | "halted";

export type AgentStatus = "idle" | "planning" | "executing" | "done" | "error" | "halted";

export interface ToolParameter {
  name: string;
  type: "string" | "number" | "boolean" | "object";
  description: string;
  required?: boolean;
}

export interface ToolDefinition {
  /** Unique tool identifier */
  name: string;
  /** Short description for the LLM to understand when to use this tool */
  description: string;
  /** Parameter schema */
  parameters: ToolParameter[];
  /** Execute the tool with the given arguments. Returns a string result. */
  execute: (args: Record<string, unknown>) => Promise<string>;
}

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
}

export interface ReasoningStep {
  id: string;
  kind: StepKind;
  content: string;
  toolCall?: ToolCall;
  durationMs?: number;
  timestamp: number;
}

export interface AgentRun {
  input: string;
  steps: ReasoningStep[];
  answer: string;
  status: AgentStatus;
  totalDurationMs: number;
  error?: string;
  /** Present only when status === "halted". */
  haltReason?: HaltReason;
}

export interface ToolPilotConfig {
  /** LLM configuration */
  llm: LLMConfig;
  /** Tools available to the agent */
  tools: ToolDefinition[];
  /** System prompt prepended to every run. Defaults to a built-in ReAct prompt. */
  systemPrompt?: string;
  /** Maximum reasoning steps before the agent is forced to answer. Default: 10 */
  maxSteps?: number;
  /** Opt-in loop governance: budget ceilings, convergence/drift detection, cancellation. Off by default. */
  guard?: LoopGuardConfig;
}

export interface UseToolPilotOptions {
  config: ToolPilotConfig;
  onStep?: (step: ReasoningStep) => void;
  onComplete?: (run: AgentRun) => void;
  onError?: (error: Error) => void;
}

export interface UseToolPilotReturn {
  run: (input: string) => Promise<void>;
  steps: ReasoningStep[];
  answer: string;
  status: AgentStatus;
  error: Error | null;
  reset: () => void;
}

export interface ToolPilotProps {
  config: ToolPilotConfig;
  input?: string;
  placeholder?: string;
  onComplete?: (run: AgentRun) => void;
  onError?: (error: Error) => void;
  theme?: "dark" | "light";
}
