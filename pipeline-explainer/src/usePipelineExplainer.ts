import { useCallback, useRef, useState } from "react";
import { getAdapter, type LLMConfig } from "./adapters";
import { buildPipelinePrompt, parsePipelineNarration } from "./prompt";
import type { Pipeline, PipelineNarration } from "./types";

export interface ExplainerState {
  narration: PipelineNarration | null;
  loading: boolean;
  error: string | null;
}

export function usePipelineExplainer(llm: LLMConfig) {
  const [state, setState] = useState<ExplainerState>({ narration: null, loading: false, error: null });
  const requestId = useRef(0);

  const explain = useCallback(
    async (pipeline: Pipeline, context?: string): Promise<PipelineNarration | null> => {
      const id = ++requestId.current;
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const adapter = getAdapter(llm);
        const raw = await adapter.complete(buildPipelinePrompt(pipeline, context), llm);
        const narration = parsePipelineNarration(raw) as PipelineNarration;
        if (id === requestId.current) setState({ narration, loading: false, error: null });
        return narration;
      } catch (err) {
        if (id === requestId.current) {
          setState({ narration: null, loading: false, error: err instanceof Error ? err.message : String(err) });
        }
        return null;
      }
    },
    [llm]
  );

  const reset = useCallback(() => setState({ narration: null, loading: false, error: null }), []);
  return { ...state, explain, reset };
}
