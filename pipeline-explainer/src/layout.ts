import type { Pipeline, PipelineNode } from "./types";

export interface PositionedNode extends PipelineNode {
  x: number;
  y: number;
  layer: number;
}

/** Simple longest-path layering: root(s) at layer 0, children below their deepest parent. */
export function layoutDag(pipeline: Pipeline): { nodes: PositionedNode[]; width: number; height: number } {
  const byId = new Map(pipeline.nodes.map((n) => [n.id, n]));
  const layerOf = new Map<string, number>();

  const layer = (id: string, seen: Set<string> = new Set()): number => {
    if (layerOf.has(id)) return layerOf.get(id)!;
    if (seen.has(id)) return 0; // cycle guard — DAGs shouldn't cycle, but don't hang
    seen.add(id);
    const node = byId.get(id);
    const parents = (node?.dependsOn ?? []).filter((p) => byId.has(p));
    const l = parents.length === 0 ? 0 : Math.max(...parents.map((p) => layer(p, seen))) + 1;
    layerOf.set(id, l);
    return l;
  };
  pipeline.nodes.forEach((n) => layer(n.id));

  const layers = new Map<number, PipelineNode[]>();
  for (const n of pipeline.nodes) {
    const l = layerOf.get(n.id) ?? 0;
    if (!layers.has(l)) layers.set(l, []);
    layers.get(l)!.push(n);
  }

  const NODE_W = 180, NODE_H = 56, GAP_X = 40, GAP_Y = 70;
  const maxPerLayer = Math.max(...[...layers.values()].map((l) => l.length));
  const width = Math.max(1, maxPerLayer) * (NODE_W + GAP_X) + GAP_X;
  const height = layers.size * (NODE_H + GAP_Y) + GAP_Y;

  const nodes: PositionedNode[] = [];
  for (const [l, ns] of [...layers.entries()].sort((a, b) => a[0] - b[0])) {
    const rowWidth = ns.length * (NODE_W + GAP_X) - GAP_X;
    const startX = (width - rowWidth) / 2;
    ns.forEach((n, i) => {
      nodes.push({ ...n, layer: l, x: startX + i * (NODE_W + GAP_X), y: GAP_Y / 2 + l * (NODE_H + GAP_Y) });
    });
  }
  return { nodes, width, height };
}

export const NODE_SIZE = { w: 180, h: 56 };
