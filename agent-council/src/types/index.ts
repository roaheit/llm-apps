export type LLMProvider = "anthropic" | "openai" | "mistral" | "custom";

export interface LLMConfig {
  provider: LLMProvider;
  apiKey: string;
  model?: string;
  maxTokens?: number;
  adapter?: (prompt: string, systemPrompt?: string) => Promise<string>;
}

export type AgentMode = "sequential" | "parallel";

export type AgentStatus = "idle" | "running" | "done" | "error";

export interface AgentConfig {
  /** Unique identifier for this agent */
  id: string;
  /** Display name shown in the UI */
  name: string;
  /** Short description of this agent's role */
  role: string;
  /** Emoji or icon to represent this agent */
  icon?: string;
  /**
   * System prompt that defines the agent's personality and task.
   * Use {{input}} to reference the original user input.
   * Use {{previous}} to reference the previous agent's output (sequential mode).
   * Use {{context}} to reference all prior agent outputs (sequential mode).
   */
  systemPrompt: string;
  /** LLM config for this agent. Falls back to pipeline-level config if omitted. */
  llm?: LLMConfig;
}

export interface SynthesizerConfig {
  /** Display name for the synthesizer */
  name?: string;
  /** Icon for the synthesizer */
  icon?: string;
  /**
   * System prompt for the synthesizer.
   * Use {{input}} for original input, {{outputs}} for all agent outputs as structured text.
   */
  systemPrompt: string;
  /** LLM config for the synthesizer. Falls back to pipeline-level config if omitted. */
  llm?: LLMConfig;
}

export interface PipelineConfig {
  /** How agents run — sequential or parallel */
  mode: AgentMode;
  /** The agents to run */
  agents: AgentConfig[];
  /** The final synthesizer that combines all agent outputs */
  synthesizer: SynthesizerConfig;
  /** Fallback LLM config used by any agent that doesn't define its own */
  llm: LLMConfig;
}

export interface AgentResult {
  agentId: string;
  agentName: string;
  agentRole: string;
  icon?: string;
  output: string;
  status: AgentStatus;
  durationMs?: number;
  error?: string;
}

export interface PipelineResult {
  agentResults: AgentResult[];
  synthesis: string;
  totalDurationMs: number;
}

export interface UseMultiAgentOptions {
  pipeline: PipelineConfig;
  onAgentComplete?: (result: AgentResult) => void;
  onComplete?: (result: PipelineResult) => void;
  onError?: (error: Error) => void;
}

export interface UseMultiAgentReturn {
  run: (input: string) => Promise<void>;
  agentResults: AgentResult[];
  synthesis: string;
  running: boolean;
  activeAgentId: string | null;
  error: Error | null;
  reset: () => void;
}

export interface MultiAgentReasoningProps {
  pipeline: PipelineConfig;
  input?: string;
  placeholder?: string;
  onComplete?: (result: PipelineResult) => void;
  onError?: (error: Error) => void;
  theme?: "dark" | "light";
  className?: string;
  style?: React.CSSProperties;
}
