import React, { useState } from "react";
import { useMultiAgent } from "../hooks/useMultiAgent";
import type { MultiAgentReasoningProps, AgentResult } from "../types";

function AgentCard({
  result,
  isActive,
  index,
  isDark,
}: {
  result: AgentResult;
  isActive: boolean;
  index: number;
  isDark: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

  const C = {
    surface:    isDark ? "#0d1424" : "#ffffff",
    surfaceAlt: isDark ? "#0a1020" : "#f1f3f7",
    border:     isDark ? "#ffffff0d" : "#e2e8f0",
    text:       isDark ? "#e2e8f0" : "#1e293b",
    muted:      "#64748b",
    accent:     "#00ffe7",
    purple:     "#a78bfa",
    error:      "#f87171",
  };

  const statusColor = result.status === "error" ? C.error
    : result.status === "done"  ? C.accent
    : C.purple;

  return (
    <div style={{
      display: "flex",
      gap: "16px",
      animation: "mar-fadein .4s ease forwards",
    }}>
      {/* Timeline spine */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: "32px" }}>
        <div style={{
          width: "32px", height: "32px", borderRadius: "50%",
          background: result.status === "error" ? "#f8717120" : "#00ffe715",
          border: `2px solid ${statusColor}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "14px", flexShrink: 0,
          boxShadow: isActive ? `0 0 16px ${statusColor}60` : "none",
          transition: "box-shadow .3s",
        }}>
          {result.icon ?? "🤖"}
        </div>
        <div style={{ width: "2px", flex: 1, background: isDark ? "#ffffff08" : "#00000008", marginTop: "6px" }}/>
      </div>

      {/* Card */}
      <div style={{
        flex: 1,
        marginBottom: "20px",
        background: C.surface,
        border: `1px solid ${isActive ? statusColor + "40" : C.border}`,
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: isActive ? `0 0 24px ${statusColor}18` : "none",
        transition: "box-shadow .3s, border-color .3s",
      }}>
        {/* Card header */}
        <div
          onClick={() => setExpanded(e => !e)}
          style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            padding: "12px 16px",
            background: C.surfaceAlt,
            borderBottom: expanded ? `1px solid ${C.border}` : "none",
            cursor: "pointer", userSelect: "none",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div>
              <div style={{ fontSize: "13px", fontWeight: 600, color: C.text }}>{result.agentName}</div>
              <div style={{ fontSize: "11px", color: C.muted }}>{result.agentRole}</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {result.durationMs && (
              <span style={{ fontSize: "11px", color: C.muted, fontFamily: "monospace" }}>
                {(result.durationMs / 1000).toFixed(1)}s
              </span>
            )}
            <div style={{
              width: "7px", height: "7px", borderRadius: "50%",
              background: statusColor,
              boxShadow: `0 0 8px ${statusColor}`,
            }}/>
            <span style={{ fontSize: "11px", color: C.muted }}>{expanded ? "▲" : "▼"}</span>
          </div>
        </div>

        {/* Card body */}
        {expanded && (
          <div style={{ padding: "16px" }}>
            {result.status === "error" ? (
              <p style={{ color: C.error, fontSize: "13px", margin: 0, fontFamily: "monospace" }}>
                ✗ {result.error}
              </p>
            ) : (
              <p style={{
                fontFamily: "'Georgia', serif",
                fontSize: "15px",
                lineHeight: 1.75,
                color: C.text,
                margin: 0,
                whiteSpace: "pre-wrap",
              }}>
                {result.output}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function SynthesisCard({ synthesis, name, icon, isDark }: {
  synthesis: string;
  name: string;
  icon: string;
  isDark: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const C = {
    text: isDark ? "#e2e8f0" : "#1e293b",
    muted: "#64748b",
  };

  const copy = () => {
    navigator.clipboard.writeText(synthesis);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{
      display: "flex", gap: "16px",
      animation: "mar-fadein .4s ease forwards",
    }}>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: "32px" }}>
        <div style={{
          width: "32px", height: "32px", borderRadius: "50%",
          background: "linear-gradient(135deg, #00ffe720, #a78bfa20)",
          border: "2px solid #a78bfa",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "14px", flexShrink: 0,
          boxShadow: "0 0 16px #a78bfa40",
        }}>
          {icon}
        </div>
      </div>
      <div style={{
        flex: 1,
        background: isDark ? "#0d1424" : "#ffffff",
        border: "1px solid #a78bfa40",
        borderRadius: "12px",
        overflow: "hidden",
        boxShadow: "0 0 32px #a78bfa18",
      }}>
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "12px 16px",
          background: isDark ? "#0f0d1f" : "#f5f3ff",
          borderBottom: `1px solid #a78bfa20`,
        }}>
          <div>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "#a78bfa" }}>{name}</div>
            <div style={{ fontSize: "11px", color: C.muted }}>Final synthesis</div>
          </div>
          <button
            onClick={copy}
            style={{
              cursor: "pointer", border: "1px solid #a78bfa30", background: "transparent",
              color: C.muted, fontSize: "12px", padding: "4px 12px", borderRadius: "6px",
              fontFamily: "inherit", transition: "all .18s",
            }}
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <div style={{ padding: "20px" }}>
          <p style={{
            fontFamily: "'Georgia', serif",
            fontSize: "16px",
            lineHeight: 1.8,
            color: C.text,
            margin: 0,
            whiteSpace: "pre-wrap",
          }}>
            {synthesis}
          </p>
        </div>
        <div style={{ height: "2px", background: "linear-gradient(90deg, #00ffe730, #a78bfa60, #00ffe730)" }}/>
      </div>
    </div>
  );
}

function ActivePulse({ label, icon, isDark }: { label: string; icon: string; isDark: boolean }) {
  return (
    <div style={{
      display: "flex", gap: "16px", alignItems: "center",
      animation: "mar-fadein .3s ease forwards",
    }}>
      <div style={{ minWidth: "32px", display: "flex", justifyContent: "center" }}>
        <div style={{
          width: "32px", height: "32px", borderRadius: "50%",
          background: "#00ffe715",
          border: "2px solid #00ffe7",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "14px",
          animation: "mar-pulse 1.2s ease-in-out infinite",
        }}>
          {icon}
        </div>
      </div>
      <div style={{
        flex: 1, padding: "12px 16px",
        background: isDark ? "#0d1424" : "#ffffff",
        border: "1px solid #00ffe730",
        borderRadius: "12px",
        display: "flex", alignItems: "center", gap: "12px",
      }}>
        <div style={{ display: "flex", gap: "4px" }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: "5px", height: "5px", borderRadius: "50%",
              background: "#00ffe7",
              animation: `mar-bounce 1s ease-in-out ${i * 0.15}s infinite`,
            }}/>
          ))}
        </div>
        <span style={{ fontSize: "13px", color: isDark ? "#64748b" : "#94a3b8" }}>{label} is reasoning…</span>
      </div>
    </div>
  );
}

export function MultiAgentReasoning({
  pipeline,
  input: inputProp = "",
  placeholder = "Describe a problem, question, or decision for the agents to reason through…",
  onComplete,
  onError,
  theme = "dark",
  className,
  style,
}: MultiAgentReasoningProps) {
  const isDark = theme === "dark";
  const [input, setInput] = useState(inputProp);

  const { run, agentResults, synthesis, running, activeAgentId, error, reset } = useMultiAgent({
    pipeline,
    onComplete,
    onError,
  });

  const C = {
    bg:         isDark ? "#080c14" : "#f8f9fb",
    surface:    isDark ? "#0d1424" : "#ffffff",
    surfaceAlt: isDark ? "#0a1020" : "#f1f3f7",
    border:     isDark ? "#ffffff0d" : "#e2e8f0",
    text:       isDark ? "#e2e8f0"  : "#1e293b",
    muted:      "#64748b",
    accent:     "#00ffe7",
    purple:     "#a78bfa",
    error:      "#f87171",
  };

  const activeAgent = pipeline.agents.find(a => a.id === activeAgentId);
  const synthesizerActive = activeAgentId === "__synthesizer__";
  const hasResults = agentResults.length > 0 || synthesis;

  return (
    <div className={className} style={{
      background: C.bg,
      fontFamily: "'DM Sans', system-ui, sans-serif",
      color: C.text,
      borderRadius: "16px",
      padding: "32px",
      maxWidth: "820px",
      width: "100%",
      display: "grid",
      gap: "24px",
      border: `1px solid ${C.border}`,
      ...style,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box}
        .mar-ta{width:100%;background:transparent;border:none;outline:none;resize:none;font-family:'DM Sans',sans-serif;font-size:15px;line-height:1.6;color:${C.text};caret-color:#00ffe7;padding:0;min-height:80px;}
        .mar-btn{cursor:pointer;border:none;background:linear-gradient(135deg,#00ffe7,#a78bfa);color:#080c14;font-family:'DM Sans',sans-serif;font-size:14px;font-weight:600;padding:11px 26px;border-radius:10px;transition:all 0.18s;letter-spacing:.02em;}
        .mar-btn:hover:not(:disabled){opacity:.88;transform:translateY(-1px);box-shadow:0 8px 24px #00ffe728;}
        .mar-btn:disabled{opacity:.38;cursor:not-allowed;}
        .mar-reset{cursor:pointer;border:1px solid ${C.border};background:transparent;color:${C.muted};font-family:'DM Sans',sans-serif;font-size:13px;padding:10px 18px;border-radius:10px;transition:all .18s;}
        .mar-reset:hover{border-color:${C.accent}40;color:${C.text};}
        @keyframes mar-fadein{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes mar-pulse{0%,100%{box-shadow:0 0 0 0 #00ffe740}50%{box-shadow:0 0 0 8px #00ffe700}}
        @keyframes mar-bounce{0%,100%{transform:translateY(0)}50%{transform:translateY(-4px)}}
        @keyframes mar-sh{0%{background-position:-200% center}100%{background-position:200% center}}
        .mar-bar{height:2px;background:linear-gradient(90deg,transparent,#00ffe7,#a78bfa,transparent);background-size:200% auto;animation:mar-sh 1.4s linear infinite;border-radius:1px;}
        .mar-mode{display:inline-flex;align-items:center;gap:5px;background:#00ffe718;border:1px solid #00ffe740;border-radius:5px;padding:2px 9px;font-size:11px;color:#00ffe7;font-family:'JetBrains Mono',monospace;}
      `}</style>

      {/* Header */}
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "8px" }}>
          <h2 style={{
            fontFamily: "system-ui", fontSize: "22px", fontWeight: 700,
            margin: 0, color: C.text, letterSpacing: "-0.02em",
          }}>
            Multi-Agent Reasoning
          </h2>
          <span className="mar-mode">{pipeline.mode}</span>
        </div>
        <p style={{ color: C.muted, fontSize: "13px", margin: 0 }}>
          {pipeline.agents.length} agents · {pipeline.synthesizer.name ?? "Synthesizer"} · {pipeline.mode === "sequential" ? "agents share context" : "agents run in parallel"}
        </p>
      </div>

      {/* Agent roster */}
      <div style={{
        display: "flex", gap: "8px", flexWrap: "wrap",
        padding: "14px", background: C.surface,
        border: `1px solid ${C.border}`, borderRadius: "12px",
      }}>
        {pipeline.agents.map((agent, i) => (
          <div key={agent.id} style={{
            display: "flex", alignItems: "center", gap: "7px",
            padding: "6px 12px",
            background: C.surfaceAlt,
            border: `1px solid ${activeAgentId === agent.id ? C.accent + "60" : C.border}`,
            borderRadius: "8px",
            transition: "border-color .3s",
          }}>
            <span style={{ fontSize: "13px" }}>{agent.icon ?? "🤖"}</span>
            <div>
              <div style={{ fontSize: "12px", fontWeight: 500, color: C.text }}>{agent.name}</div>
              <div style={{ fontSize: "10px", color: C.muted }}>{agent.role}</div>
            </div>
            {i < pipeline.agents.length - 1 && pipeline.mode === "sequential" && (
              <span style={{ marginLeft: "4px", color: C.muted, fontSize: "11px" }}>→</span>
            )}
          </div>
        ))}
        <div style={{ display:"flex", alignItems:"center", gap:"7px", padding:"6px 12px", background: C.surfaceAlt, border:`1px solid ${synthesizerActive ? "#a78bfa60" : C.border}`, borderRadius:"8px", transition:"border-color .3s" }}>
          <span style={{ fontSize:"13px" }}>{pipeline.synthesizer.icon ?? "⚡"}</span>
          <div>
            <div style={{ fontSize:"12px", fontWeight:500, color:"#a78bfa" }}>{pipeline.synthesizer.name ?? "Synthesizer"}</div>
            <div style={{ fontSize:"10px", color:C.muted }}>final synthesis</div>
          </div>
        </div>
      </div>

      {/* Input */}
      <div style={{
        background: C.surface, border: `1px solid ${C.border}`,
        borderRadius: "12px", overflow: "hidden",
      }}>
        <div style={{ padding: "14px 16px 0" }}>
          <textarea
            className="mar-ta"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={placeholder}
            disabled={running}
          />
        </div>
        <div style={{ padding: "10px 16px 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "12px", color: C.muted }}>{input.length} chars</span>
          <div style={{ display: "flex", gap: "10px" }}>
            {hasResults && !running && (
              <button className="mar-reset" onClick={() => { reset(); setInput(""); }}>Reset</button>
            )}
            <button
              className="mar-btn"
              onClick={() => run(input)}
              disabled={running || !input.trim()}
            >
              {running ? "Reasoning…" : "Run Pipeline →"}
            </button>
          </div>
        </div>
      </div>

      {/* Progress bar */}
      {running && <div className="mar-bar" />}

      {/* Error */}
      {error && (
        <div style={{ background: "#f8717110", border: "1px solid #f8717130", borderRadius: "10px", padding: "11px 15px", color: C.error, fontSize: "14px" }}>
          {error.message}
        </div>
      )}

      {/* Timeline */}
      {hasResults && (
        <div>
          <p style={{ fontSize: "11px", color: C.muted, letterSpacing: ".1em", textTransform: "uppercase", margin: "0 0 16px", fontFamily: "'JetBrains Mono',monospace" }}>
            Reasoning Trace
          </p>
          <div>
            {agentResults.map((result, i) => (
              <AgentCard
                key={result.agentId}
                result={result}
                isActive={activeAgentId === result.agentId}
                index={i}
                isDark={isDark}
              />
            ))}

            {/* Active agent pulse */}
            {running && activeAgent && (
              <ActivePulse
                label={activeAgent.name}
                icon={activeAgent.icon ?? "🤖"}
                isDark={isDark}
              />
            )}

            {/* Synthesizer pulse */}
            {synthesizerActive && (
              <ActivePulse
                label={pipeline.synthesizer.name ?? "Synthesizer"}
                icon={pipeline.synthesizer.icon ?? "⚡"}
                isDark={isDark}
              />
            )}

            {/* Synthesis output */}
            {synthesis && (
              <SynthesisCard
                synthesis={synthesis}
                name={pipeline.synthesizer.name ?? "Synthesizer"}
                icon={pipeline.synthesizer.icon ?? "⚡"}
                isDark={isDark}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
