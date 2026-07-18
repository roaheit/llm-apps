import { useState, useCallback } from "react";
import { runAgent } from "../planner";
import type {
  UseToolPilotOptions,
  UseToolPilotReturn,
  ReasoningStep,
  AgentStatus,
} from "../types";

export function useToolPilot({
  config,
  onStep,
  onComplete,
  onError,
}: UseToolPilotOptions): UseToolPilotReturn {
  const [steps, setSteps]     = useState<ReasoningStep[]>([]);
  const [answer, setAnswer]   = useState("");
  const [status, setStatus]   = useState<AgentStatus>("idle");
  const [error, setError]     = useState<Error | null>(null);

  const reset = useCallback(() => {
    setSteps([]);
    setAnswer("");
    setStatus("idle");
    setError(null);
  }, []);

  const run = useCallback(async (input: string) => {
    reset();
    setStatus("planning");

    try {
      const result = await runAgent(input, config, (step) => {
        setSteps((prev) => [...prev, step]);
        if (step.kind === "tool-call") setStatus("executing");
        onStep?.(step);
      });

      setAnswer(result.answer);
      setStatus(result.status);

      if (result.error) {
        const err = new Error(result.error);
        setError(err);
        onError?.(err);
      } else {
        onComplete?.(result);
      }
    } catch (e) {
      const err = e as Error;
      setError(err);
      setStatus("error");
      onError?.(err);
    }
  }, [config, onStep, onComplete, onError, reset]);

  return { run, steps, answer, status, error, reset };
}
