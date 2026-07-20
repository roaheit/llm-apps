import { useEffect, useMemo } from "react";
import { useSqlNarrator } from "./useSqlNarrator";
import type { LLMConfig, NarrationRequest } from "./types";

export interface SqlNarratorProps extends NarrationRequest {
  llm: LLMConfig;
  /** Re-narrate automatically when sql/results change. Default true. */
  auto?: boolean;
  /** Override any of the default section renderers via className hooks. */
  className?: string;
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    fontFamily: "ui-sans-serif, system-ui, sans-serif",
    border: "1px solid #d8d4cc",
    borderRadius: 10,
    overflow: "hidden",
    background: "#fbfaf7",
    color: "#25231f",
  },
  header: {
    display: "flex",
    alignItems: "baseline",
    gap: 10,
    padding: "12px 16px",
    borderBottom: "1px solid #e4e0d8",
    background: "#f2efe8",
  },
  body: { padding: 16, lineHeight: 1.55, fontSize: 14 },
  summary: { fontSize: 15, fontWeight: 600, marginBottom: 12 },
  sectionLabel: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.08em",
    textTransform: "uppercase" as const,
    color: "#8a8577",
    margin: "14px 0 4px",
  },
  caveat: {
    margin: "6px 0 0",
    padding: "8px 10px",
    background: "#fdf6e3",
    border: "1px solid #ead9a8",
    borderRadius: 6,
    fontSize: 13,
  },
  state: { padding: 16, fontSize: 14, color: "#6b6659" },
  error: { padding: 16, fontSize: 14, color: "#8c2f24", background: "#fbeeec" },
};

export function SqlNarrator({ llm, auto = true, className, ...request }: SqlNarratorProps) {
  const { narration, loading, error, narrate } = useSqlNarrator(llm);

  const requestKey = useMemo(
    () => JSON.stringify([request.sql, request.results, request.tone, request.dialect]),
    [request.sql, request.results, request.tone, request.dialect]
  );

  useEffect(() => {
    if (auto && request.sql?.trim()) {
      narrate(request);
    }
  }, [auto, requestKey, narrate]);

  return (
    <div style={styles.root} className={className}>
      <div style={styles.header}>
        <span style={{ fontWeight: 700, fontSize: 14 }}>SQL Narrator</span>
        <span style={{ fontSize: 12, color: "#8a8577" }}>
          {request.dialect ?? "ansi"} · {request.tone ?? "analyst"}
        </span>
      </div>
      {loading && <div style={styles.state}>Reading the query…</div>}
      {error && <div style={styles.error}>Couldn’t narrate this query: {error}</div>}
      {!loading && !error && narration && (
        <div style={styles.body}>
          <div style={styles.summary}>{narration.summary}</div>
          <div style={styles.sectionLabel}>What the query does</div>
          <div>{narration.queryExplanation}</div>
          {narration.resultsInterpretation && (
            <>
              <div style={styles.sectionLabel}>What the results mean</div>
              <div>{narration.resultsInterpretation}</div>
            </>
          )}
          {narration.caveats && narration.caveats.length > 0 && (
            <>
              <div style={styles.sectionLabel}>Caveats</div>
              {narration.caveats.map((c, i) => (
                <div key={i} style={styles.caveat}>{c}</div>
              ))}
            </>
          )}
        </div>
      )}
      {!loading && !error && !narration && (
        <div style={styles.state}>Provide a SQL query to get a plain-English narration.</div>
      )}
    </div>
  );
}
