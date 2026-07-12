import type { Pipeline, PipelineNode } from "../types";

/**
 * Parse Snowflake CREATE TASK DDL into a Pipeline.
 * Handles: CREATE [OR REPLACE] TASK <name>, AFTER <a>, <b>, SCHEDULE = '...',
 * WAREHOUSE = X, WHEN <condition>, FINALIZE = <root>, and the AS <sql body>.
 * Pragmatic regex parsing — not a full SQL parser; covers standard task DDL scripts.
 */
export function parseSnowflakeTasks(ddl: string): Pipeline {
  const nodes: PipelineNode[] = [];
  // Split on CREATE ... TASK boundaries, keeping each statement.
  const stmts = ddl.split(/(?=create\s+(?:or\s+replace\s+)?task\s)/i).filter((s) => /create\s+(?:or\s+replace\s+)?task/i.test(s));

  for (const stmt of stmts) {
    const nameMatch = stmt.match(/create\s+(?:or\s+replace\s+)?task\s+(?:if\s+not\s+exists\s+)?([\w."]+)/i);
    if (!nameMatch) continue;
    const id = nameMatch[1].replace(/"/g, "");

    const afterMatch = stmt.match(/\bafter\s+([\w.",\s]+?)(?=\b(?:when|as|schedule|warehouse|finalize)\b|$)/i);
    const dependsOn = afterMatch
      ? afterMatch[1].split(",").map((t) => t.trim().replace(/"/g, "")).filter(Boolean)
      : [];

    const scheduleMatch = stmt.match(/schedule\s*=\s*'([^']+)'/i);
    const warehouseMatch = stmt.match(/warehouse\s*=\s*([\w"]+)/i);
    const whenMatch = stmt.match(/\bwhen\s+([\s\S]+?)\s+as\s/i);
    const finalizeMatch = stmt.match(/finalize\s*=\s*([\w."]+)/i);
    const bodyMatch = stmt.match(/\bas\s+([\s\S]+?);?\s*$/i);

    const node: PipelineNode = {
      id,
      dependsOn: finalizeMatch ? [finalizeMatch[1].replace(/"/g, "")] : dependsOn,
      finalizer: Boolean(finalizeMatch),
    };
    if (scheduleMatch) node.schedule = scheduleMatch[1];
    if (warehouseMatch) node.warehouse = warehouseMatch[1].replace(/"/g, "");
    if (whenMatch) node.condition = whenMatch[1].trim();
    if (bodyMatch) node.body = bodyMatch[1].trim();
    nodes.push(node);
  }

  return { nodes };
}
