export interface ResolvedConvergenceConfig {
  minConsecutive: number;
  similarityThreshold: number;
  similarityFn: (a: string, b: string) => number;
}

export interface ConvergenceResult {
  converged: boolean;
}

/**
 * "Is this one state stuck?" — the last `minConsecutive` entries in `history`
 * must all be pairwise-similar (each adjacent pair clears `similarityThreshold`)
 * before declaring convergence. Requiring several consecutive matches (not
 * just one) distinguishes a genuine plateau from a coincidental one-off repeat.
 */
export function detectConvergence(
  history: string[],
  cfg: ResolvedConvergenceConfig
): ConvergenceResult {
  if (history.length < cfg.minConsecutive) return { converged: false };

  const recent = history.slice(-cfg.minConsecutive);
  for (let i = 1; i < recent.length; i++) {
    if (cfg.similarityFn(recent[i], recent[i - 1]) < cfg.similarityThreshold) {
      return { converged: false };
    }
  }
  return { converged: true };
}
