import { useCallback, useRef, useState } from "react";
import { complete } from "llm-core";
import { buildPrompt, parseNarration } from "./prompt";
import type { LLMConfig, Narration, NarrationRequest, NarratorState } from "./types";

export interface UseSqlNarratorReturn extends NarratorState {
  narrate: (req: NarrationRequest) => Promise<Narration | null>;
  reset: () => void;
}

export function useSqlNarrator(llm: LLMConfig): UseSqlNarratorReturn {
  const [state, setState] = useState<NarratorState>({
    narration: null,
    loading: false,
    error: null,
  });
  const requestId = useRef(0);

  const narrate = useCallback(
    async (req: NarrationRequest): Promise<Narration | null> => {
      const id = ++requestId.current;
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const { text: raw } = await complete(llm, { prompt: buildPrompt(req), responseFormat: "json" });
        const narration = parseNarration(raw) as Narration;
        if (id === requestId.current) {
          setState({ narration, loading: false, error: null });
        }
        return narration;
      } catch (err) {
        if (id === requestId.current) {
          setState({
            narration: null,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
          });
        }
        return null;
      }
    },
    [llm]
  );

  const reset = useCallback(
    () => setState({ narration: null, loading: false, error: null }),
    []
  );

  return { ...state, narrate, reset };
}
