import { useMemo } from "react";
import { layoutDag, NODE_SIZE } from "./layout";
import type { Pipeline } from "./types";

export interface DagViewProps {
  pipeline: Pipeline;
  /** Node id → note shown as tooltip (from narration.nodeNotes). */
  notes?: Record<string, string>;
  /** Highlight a node id. */
  selected?: string | null;
  onSelect?: (id: string) => void;
}

export function DagView({ pipeline, notes, selected, onSelect }: DagViewProps) {
  const { nodes, width, height } = useMemo(() => layoutDag(pipeline), [pipeline]);
  const pos = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const { w, h } = NODE_SIZE;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", background: "#101418", borderRadius: 10 }}
      role="img"
      aria-label={`Pipeline DAG with ${nodes.length} nodes`}
    >
      <defs>
        <marker id="pe-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 8 4 L 0 8 z" fill="#5f6b78" />
        </marker>
      </defs>
      {nodes.flatMap((n) =>
        n.dependsOn
          .filter((p) => pos.has(p))
          .map((p) => {
            const parent = pos.get(p)!;
            return (
              <path
                key={`${p}->${n.id}`}
                d={`M ${parent.x + w / 2} ${parent.y + h} C ${parent.x + w / 2} ${parent.y + h + 30}, ${n.x + w / 2} ${n.y - 30}, ${n.x + w / 2} ${n.y}`}
                fill="none"
                stroke={n.finalizer ? "#8a6d3b" : "#5f6b78"}
                strokeWidth={1.5}
                strokeDasharray={n.finalizer ? "5 4" : undefined}
                markerEnd="url(#pe-arrow)"
              />
            );
          })
      )}
      {nodes.map((n) => {
        const isRoot = n.dependsOn.length === 0 && !n.finalizer;
        const stroke = n.id === selected ? "#e8c268" : n.finalizer ? "#8a6d3b" : isRoot ? "#4f9d69" : "#3a4652";
        return (
          <g key={n.id} onClick={() => onSelect?.(n.id)} style={{ cursor: onSelect ? "pointer" : "default" }}>
            <rect x={n.x} y={n.y} width={w} height={h} rx={8} fill="#1a2027" stroke={stroke} strokeWidth={n.id === selected ? 2 : 1.25} />
            <text x={n.x + 10} y={n.y + 22} fill="#e6e9ec" fontSize="13" fontFamily="ui-monospace, monospace">
              {n.id.length > 22 ? n.id.slice(0, 21) + "…" : n.id}
            </text>
            <text x={n.x + 10} y={n.y + 42} fill="#8b96a1" fontSize="11" fontFamily="ui-sans-serif, sans-serif">
              {n.finalizer ? "finalizer" : isRoot ? (n.schedule ? `⏱ ${n.schedule.slice(0, 20)}` : "root") : n.condition ? "conditional" : `after ${n.dependsOn.length}`}
            </text>
            {notes?.[n.id] && <title>{notes[n.id]}</title>}
          </g>
        );
      })}
    </svg>
  );
}
