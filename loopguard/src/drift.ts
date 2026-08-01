export interface ResolvedDriftConfig {
  minCycleLength: number;
  maxCycleLength: number;
  minRepeats: number;
  similarityThreshold: number;
  similarityFloor: number;
  similarityFn: (a: string, b: string) => number;
}

export interface DriftResult {
  drift: boolean;
  period?: number;
}

/**
 * "Is this cycling between >=2 states?" — for each candidate cycle length `L`
 * from `minCycleLength` to `maxCycleLength` (shortest wins), compares
 * `history[i]` against `history[i-L]` across the required window. Uses an
 * AVERAGE similarity across the cycle's pairs (not "every position must
 * match") with a per-pair floor: this tolerates the common real case where a
 * repeating tool-call/observation pair is near-verbatim each lap while
 * accompanying "thinking" text is legitimately reworded each time —
 * requiring every position to match individually would false-negative on
 * exactly that pattern. The floor stops one wildly-different position from
 * masking a genuine cycle in the other positions.
 */
export function detectDrift(history: string[], cfg: ResolvedDriftConfig): DriftResult {
  for (let period = cfg.minCycleLength; period <= cfg.maxCycleLength; period++) {
    const needed = period * (cfg.minRepeats + 1);
    if (history.length < needed) continue;

    const recent = history.slice(-needed);
    const scores: number[] = [];
    let floorOk = true;
    for (let i = period; i < recent.length; i++) {
      const score = cfg.similarityFn(recent[i], recent[i - period]);
      if (score < cfg.similarityFloor) {
        floorOk = false;
        break;
      }
      scores.push(score);
    }
    if (!floorOk) continue;

    const avg = scores.reduce((sum, s) => sum + s, 0) / scores.length;
    if (avg >= cfg.similarityThreshold) {
      return { drift: true, period };
    }
  }
  return { drift: false };
}
