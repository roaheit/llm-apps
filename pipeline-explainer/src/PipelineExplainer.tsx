import { useEffect, useMemo, useState } from "react";
import type { LLMConfig } from "./types";
import { DagView } from "./DagView";
import { parseSnowflakeTasks, fromJson } from "./parsers";
import type { Pipeline } from "./types";
import { usePipelineExplainer } from "./usePipelineExplainer";

export interface PipelineExplainerProps {
  llm: LLMConfig;
  /** Snowflake CREATE TASK DDL script. Provide this OR `pipeline`. */
  ddl?: string;
  /** Already-structured DAG. Provide this OR `ddl`. */
  pipeline?: Pipeline;
  /** Business context passed to the LLM. */
  context?: string;
  /** Auto-explain on mount / input change. Default true. */
  auto?: boolean;
  className?: string;
}

const S: Record<string, React.CSSProperties> = {
  root: { fontFamily: "ui-sans-serif, system-ui, sans-serif", color: "#25231f" },
  panel: { border: "1px solid #d8d4cc", borderRadius: 10, background: "#fbfaf7", padding: 16, marginTop: 12, lineHeight: 1.55, fontSize: 14 },
  label: { fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" as const, color: "#8a8577", margin: "14px 0 4px" },
  summary: { fontSize: 15, fontWeight: 600, marginBottom: 10 },
  obs: { margin: "6px 0 0", padding: "8px 10px", background: "#fdf6e3", border: "1px solid #ead9a8", borderRadius: 6, fontSize: 13 },
  state: { padding: 12, fontSize: 14, color: "#6b6659" },
  error: { padding: 12, fontSize: 14, color: "#8c2f24", background: "#fbeeec", borderRadius: 8, marginTop: 12 },
  nodeNote: { padding: "10px 12px", background: "#f2efe8", borderRadius: 8, marginTop: 10, fontSize: 13 },
};

export function PipelineExplainer({ llm, ddl, pipeline, context, auto = true, className }: PipelineExplainerProps) {
  const parsed = useMemo<Pipeline | null>(() => {
    try {
      if (pipeline) return fromJson(pipeline);
      if (ddl?.trim()) return parseSnowflakeTasks(ddl);
      return null;
    } catch {
      return null;
    }
  }, [ddl, pipeline]);

  const { narration, loading, error, explain } = usePipelineExplainer(llm);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (auto && parsed && parsed.nodes.length > 0) explain(parsed, context);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auto, JSON.stringify(parsed), context]);

  if (!parsed || parsed.nodes.length === 0) {
    return <div style={S.state}>Provide Snowflake task DDL or a pipeline object to visualize.</div>;
  }

  return (
    <div style={S.root} className={className}>
      <DagView pipeline={parsed} notes={narration?.nodeNotes} selected={selected} onSelect={setSelected} />
      {selected && narration?.nodeNotes?.[selected] && (
        <div style={S.nodeNote}>
          <strong style={{ fontFamily: "ui-monospace, monospace" }}>{selected}</strong> — {narration.nodeNotes[selected]}
        </div>
      )}
      {loading && <div style={S.state}>Reading the pipeline…</div>}
      {error && <div style={S.error}>Couldn’t explain this pipeline: {error}</div>}
      {!loading && !error && narration && (
        <div style={S.panel}>
          <div style={S.summary}>{narration.summary}</div>
          <div style={S.label}>How it flows</div>
          <div>{narration.flowExplanation}</div>
          {narration.observations?.length > 0 && (
            <>
              <div style={S.label}>Observations</div>
              {narration.observations.map((o, i) => (
                <div key={i} style={S.obs}>{o}</div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
