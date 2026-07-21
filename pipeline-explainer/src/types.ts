export type { LLMConfig, LLMProvider } from "corellm";

export interface PipelineNode {
  /** Unique task/node name. */
  id: string;
  /** IDs of upstream nodes this node runs after. */
  dependsOn: string[];
  /** CRON or interval schedule — usually only on the root task. */
  schedule?: string;
  /** Conditional execution clause, e.g. SYSTEM$STREAM_HAS_DATA('S'). */
  condition?: string;
  /** The SQL body / command the node runs. */
  body?: string;
  /** Warehouse or compute hint. */
  warehouse?: string;
  /** True if this is a finalizer task. */
  finalizer?: boolean;
}

export interface Pipeline {
  name?: string;
  nodes: PipelineNode[];
}

export interface PipelineNarration {
  /** One-sentence summary of the whole pipeline. */
  summary: string;
  /** Plain-English walkthrough of the flow, in execution order. */
  flowExplanation: string;
  /** Per-node one-liners keyed by node id. */
  nodeNotes: Record<string, string>;
  /** Risks and design observations: missing finalizers, fan-in hazards, scheduling issues. */
  observations: string[];
}
