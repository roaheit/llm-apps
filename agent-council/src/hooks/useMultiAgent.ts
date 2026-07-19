import { useState, useCallback } from "react";
import { callLLM } from "llm-core";
import { interpolate, buildContext, buildOutputsSummary } from "../utils/interpolate";
import type {
  UseMultiAgentOptions,
  UseMultiAgentReturn,
  AgentResult,
  PipelineResult,
} from "../types";

export function useMultiAgent({
  pipeline,
  onAgentComplete,
  onComplete,
  onError,
}: UseMultiAgentOptions): UseMultiAgentReturn {
  const [agentResults, setAgentResults] = useState<AgentResult[]>([]);
  const [synthesis, setSynthesis]       = useState("");
  const [running, setRunning]           = useState(false);
  const [activeAgentId, setActiveAgentId] = useState<string | null>(null);
  const [error, setError]               = useState<Error | null>(null);

  const reset = useCallback(() => {
    setAgentResults([]);
    setSynthesis("");
    setError(null);
    setActiveAgentId(null);
  }, []);

  const run = useCallback(async (input: string) => {
    setRunning(true);
    setError(null);
    setAgentResults([]);
    setSynthesis("");

    const startTime = Date.now();
    const completedResults: AgentResult[] = [];

    try {
      if (pipeline.mode === "sequential") {
        // ── Sequential: each agent sees prior context ──────────────────────
        for (const agent of pipeline.agents) {
          setActiveAgentId(agent.id);
          const agentStart = Date.now();

          const llmConfig = agent.llm ?? pipeline.llm;
          const previous  = completedResults.at(-1)?.output ?? "";
          const context   = buildContext(completedResults);

          const systemPrompt = interpolate(agent.systemPrompt, { input, previous, context });
          const userPrompt   = `Input: ${input}`;

          let output = "";
          let agentError: string | undefined;

          try {
            output = await callLLM(userPrompt, systemPrompt, llmConfig);
          } catch (e) {
            agentError = (e as Error).message;
          }

          const result: AgentResult = {
            agentId:   agent.id,
            agentName: agent.name,
            agentRole: agent.role,
            icon:      agent.icon,
            output,
            status:    agentError ? "error" : "done",
            durationMs: Date.now() - agentStart,
            error:     agentError,
          };

          completedResults.push(result);
          setAgentResults(prev => [...prev, result]);
          onAgentComplete?.(result);
        }

      } else {
        // ── Parallel: all agents run simultaneously ─────────────────────────
        const agentPromises = pipeline.agents.map(async (agent) => {
          const agentStart = Date.now();
          const llmConfig  = agent.llm ?? pipeline.llm;
          const systemPrompt = interpolate(agent.systemPrompt, { input });
          const userPrompt   = `Input: ${input}`;

          let output = "";
          let agentError: string | undefined;

          try {
            output = await callLLM(userPrompt, systemPrompt, llmConfig);
          } catch (e) {
            agentError = (e as Error).message;
          }

          const result: AgentResult = {
            agentId:   agent.id,
            agentName: agent.name,
            agentRole: agent.role,
            icon:      agent.icon,
            output,
            status:    agentError ? "error" : "done",
            durationMs: Date.now() - agentStart,
            error:     agentError,
          };

          // Stream each result into UI as it completes
          setAgentResults(prev => {
            const exists = prev.find(r => r.agentId === agent.id);
            return exists ? prev.map(r => r.agentId === agent.id ? result : r) : [...prev, result];
          });
          onAgentComplete?.(result);
          return result;
        });

        const results = await Promise.all(agentPromises);
        completedResults.push(...results);
      }

      // ── Synthesizer ────────────────────────────────────────────────────────
      setActiveAgentId("__synthesizer__");
      const synthLLM     = pipeline.synthesizer.llm ?? pipeline.llm;
      const outputsSummary = buildOutputsSummary(completedResults);
      const synthSystem  = interpolate(pipeline.synthesizer.systemPrompt, {
        input,
        outputs: outputsSummary,
      });

      const synthText = await callLLM(`Input: ${input}`, synthSystem, synthLLM);
      setSynthesis(synthText);

      const finalResult: PipelineResult = {
        agentResults: completedResults,
        synthesis: synthText,
        totalDurationMs: Date.now() - startTime,
      };

      onComplete?.(finalResult);

    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      setError(err);
      onError?.(err);
    } finally {
      setRunning(false);
      setActiveAgentId(null);
    }
  }, [pipeline, onAgentComplete, onComplete, onError]);

  return { run, agentResults, synthesis, running, activeAgentId, error, reset };
}
