import React, { useState } from "react";
import { useToolPilot } from "../hooks/useToolPilot";
import type { ToolPilotProps, ReasoningStep } from "../types";

const ICONS: Record<string, string> = {
  thinking: "🧠",
  "tool-call": "🔧",
  observation: "👁",
  answer: "✅",
  error: "❌",
};

const LABELS: Record<string, string> = {
  thinking: "Thinking",
  "tool-call": "Tool Call",
  observation: "Observation",
  answer: "Answer",
  error: "Error",
};

function StepCard({
  step,
  isDark,
  isLatest,
}: {
  step: ReasoningStep;
  isDark: boolean;
  isLatest: boolean;
}) {
  const [expanded, setExpanded] = useState(true);

  const C = {
    surface: isDark ? "#0d1424" : "#ffffff",
    surfaceAlt: isDark ? "#0a1020" : "#f1f3f7",
    border: isDark ? "#ffffff0d" : "#e2e8f0",
    text: isDark ? "#e2e8f0" : "#1e293b",
    muted: "#64748b",
    accent: "#00ffe7",
    purple: "#a78bfa",
    orange: "#f59e0b",
    error: "#f87171",
    green: "#34d399",
  };

  const colorMap: Record<string, string> = {
    thinking: C.purple,
    "tool-call": C.orange,
    observation: C.accent,
    answer: C.green,
    error: C.error,
  };

  const color = colorMap[step.kind] ?? C.muted;

  return (
    <div style={{ display: "flex", gap: "16px", animation: "tp-fadein .4s ease forwards" }}>
      {/* Timeline node */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", minWidth: "32px" }}>
        <div
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "50%",
            background: `${color}15`,
            border: `2px solid ${color}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "14px",
            flexShrink: 0,
            boxShadow: isLatest ? `0 0 16px ${color}60` : "none",
            transition: "box-shadow .3s",
          }}
        >
          {ICONS[step.kind] ?? "•"}
        </div>
        <div
          style={{
            width: "2px",
            flex: 1,
            background: isDark ? "#ffffff08" : "#00000008",
            marginTop: "6px",
          }}
        />
      </div>

      {/* Card */}
      <div
        style={{
          flex: 1,
          marginBottom: "16px",
          background: C.surface,
          border: `1px solid ${isLatest ? color + "40" : C.border}`,
          borderRadius: "12px",
          overflow: "hidden",
          boxShadow: isLatest ? `0 0 24px ${color}18` : "none",
          transition: "box-shadow .3s, border-color .3s",
        }}
      >
        <div
          onClick={() => setExpanded((e) => !e)}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "10px 16px",
            background: C.surfaceAlt,
            borderBottom: expanded ? `1px solid ${C.border}` : "none",
            cursor: "pointer",
            userSelect: "none",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div
              style={{
                fontSize: "10px",
                fontFamily: "monospace",
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color,
                fontWeight: 600,
              }}
            >
              {LABELS[step.kind] ?? step.kind}
            </div>
            {step.toolCall && (
              <code
                style={{
                  fontSize: "12px",
                  color: C.text,
                  background: isDark ? "#ffffff08" : "#0000000a",
                  padding: "2px 8px",
                  borderRadius: "4px",
                }}
              >
                {step.toolCall.tool}({Object.keys(step.toolCall.args).join(", ")})
              </code>
            )}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {step.durationMs != null && (
              <span style={{ fontSize: "11px", color: C.muted, fontFamily: "monospace" }}>
                {(step.durationMs / 1000).toFixed(1)}s
              </span>
            )}
            <div
              style={{
                width: "7px",
                height: "7px",
                borderRadius: "50%",
                background: color,
                boxShadow: `0 0 8px ${color}`,
              }}
            />
            <span style={{ fontSize: "11px", color: C.muted }}>{expanded ? "▲" : "▼"}</span>
          </div>
        </div>

        {expanded && (
          <div style={{ padding: "14px 16px" }}>
            <pre
              style={{
                fontFamily: step.kind === "observation" || step.kind === "tool-call"
                  ? "'Menlo', 'Consolas', monospace"
                  : "'Georgia', serif",
                fontSize: step.kind === "answer" ? "15px" : "13px",
                lineHeight: 1.7,
                color: step.kind === "error" ? C.error : C.text,
                margin: 0,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {step.kind === "tool-call" && step.toolCall
                ? JSON.stringify(step.toolCall.args, null, 2)
                : step.content}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

function ActivePulse({ label, isDark }: { label: string; isDark: boolean }) {
  return (
    <div style={{ display: "flex", gap: "16px", alignItems: "center", animation: "tp-fadein .3s ease forwards" }}>
      <div style={{ minWidth: "32px", display: "flex", justifyContent: "center" }}>
        <div
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "50%",
            background: "#00ffe715",
            border: "2px solid #00ffe7",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "14px",
            animation: "tp-pulse 1.2s ease-in-out infinite",
          }}
        >
          ⚡
        </div>
      </div>
      <div
        style={{
          flex: 1,
          padding: "12px 16px",
          background: isDark ? "#0d1424" : "#ffffff",
          border: "1px solid #00ffe730",
          borderRadius: "12px",
          display: "flex",
          alignItems: "center",
          gap: "12px",
        }}
      >
        <div style={{ display: "flex", gap: "4px" }}>
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: "5px",
                height: "5px",
                borderRadius: "50%",
                background: "#00ffe7",
                animation: `tp-bounce 1s ease-in-out ${i * 0.15}s infinite`,
              }}
            />
          ))}
        </div>
        <span style={{ fontSize: "13px", color: isDark ? "#64748b" : "#94a3b8" }}>{label}</span>
      </div>
    </div>
  );
}

export function ToolPilot({
  config,
  input: inputProp = "",
  placeholder = "Describe a task for the agent to accomplish…",
  onComplete,
  onError,
  theme = "dark",
}: ToolPilotProps) {
  const [input, setInput] = useState(inputProp);
  const isDark = theme === "dark";

  const { run, steps, answer, status, error, reset } = useToolPilot({
    config,
    onComplete,
    onError,
  });

  const C = {
    bg: isDark ? "#080c14" : "#f8f9fb",
    surface: isDark ? "#0d1424" : "#ffffff",
    border: isDark ? "#ffffff0d" : "#e2e8f0",
    text: isDark ? "#e2e8f0" : "#1e293b",
    muted: "#64748b",
    accent: "#00ffe7",
  };

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || status === "planning" || status === "executing") return;
    run(trimmed);
  };

  const statusLabel =
    status === "planning"  ? "Agent is thinking…" :
    status === "executing" ? "Running tools…" :
    null;

  return (
    <div
      style={{
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        background: C.bg,
        color: C.text,
        minHeight: "420px",
        padding: "28px",
        borderRadius: "16px",
        border: `1px solid ${C.border}`,
      }}
    >
      <style>{`
        @keyframes tp-fadein { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes tp-pulse  { 0%, 100% { box-shadow: 0 0 0 0 #00ffe750; } 50% { box-shadow: 0 0 0 8px #00ffe700; } }
        @keyframes tp-bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }
      `}</style>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "20px" }}>
        <span style={{ fontSize: "20px" }}>🛩️</span>
        <div>
          <div style={{ fontSize: "16px", fontWeight: 700, letterSpacing: "-0.02em" }}>
            tool-pilot
          </div>
          <div style={{ fontSize: "11px", color: C.muted }}>
            {config.tools.length} tool{config.tools.length !== 1 ? "s" : ""} available
            {" · "}max {config.maxSteps ?? 10} steps
          </div>
        </div>
      </div>

      {/* Input bar */}
      <div
        style={{
          display: "flex",
          gap: "10px",
          marginBottom: "24px",
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder={placeholder}
          disabled={status === "planning" || status === "executing"}
          style={{
            flex: 1,
            padding: "12px 16px",
            fontSize: "14px",
            fontFamily: "inherit",
            background: C.surface,
            color: C.text,
            border: `1px solid ${C.border}`,
            borderRadius: "10px",
            outline: "none",
          }}
        />
        <button
          onClick={handleSubmit}
          disabled={!input.trim() || status === "planning" || status === "executing"}
          style={{
            padding: "12px 24px",
            fontSize: "13px",
            fontWeight: 600,
            fontFamily: "inherit",
            background: C.accent,
            color: "#000",
            border: "none",
            borderRadius: "10px",
            cursor:
              !input.trim() || status === "planning" || status === "executing"
                ? "not-allowed"
                : "pointer",
            opacity:
              !input.trim() || status === "planning" || status === "executing" ? 0.4 : 1,
            transition: "opacity .2s",
          }}
        >
          Run
        </button>
        {steps.length > 0 && status !== "planning" && status !== "executing" && (
          <button
            onClick={reset}
            style={{
              padding: "12px 16px",
              fontSize: "13px",
              fontFamily: "inherit",
              background: "transparent",
              color: C.muted,
              border: `1px solid ${C.border}`,
              borderRadius: "10px",
              cursor: "pointer",
            }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Reasoning trace */}
      {steps.length > 0 && (
        <div style={{ marginTop: "8px" }}>
          {steps.map((step, i) => (
            <StepCard
              key={step.id}
              step={step}
              isDark={isDark}
              isLatest={i === steps.length - 1 && (status === "planning" || status === "executing")}
            />
          ))}
        </div>
      )}

      {/* Active pulse */}
      {statusLabel && <ActivePulse label={statusLabel} isDark={isDark} />}

      {/* Error */}
      {error && status === "error" && (
        <div
          style={{
            marginTop: "16px",
            padding: "12px 16px",
            background: "#f8717115",
            border: "1px solid #f8717140",
            borderRadius: "10px",
            color: "#f87171",
            fontSize: "13px",
            fontFamily: "monospace",
          }}
        >
          ✗ {error.message}
        </div>
      )}
    </div>
  );
}
