export { parseSnowflakeTasks } from "./snowflake";
import type { Pipeline } from "../types";

/** Accept a Pipeline object directly (already-structured JSON DAGs). */
export function fromJson(input: unknown): Pipeline {
  const p = input as Pipeline;
  if (!p || !Array.isArray(p.nodes)) {
    throw new Error("fromJson expects { nodes: [{ id, dependsOn, ... }] }");
  }
  return { name: p.name, nodes: p.nodes.map((n) => ({ ...n, dependsOn: n.dependsOn ?? [] })) };
}
