import type { LLMConfig, LLMProvider, TokenUsage } from "corellm";
import type { PricingTable } from "./pricing";

export type { LLMConfig, LLMProvider, TokenUsage };

export type HaltReason =
  | "budget-tokens"
  | "budget-cost"
  | "budget-time"
  | "budget-iterations"
  | "drift"
  | "converged";

export interface LoopGuardBudgets {
  /** Hard ceiling on cumulative tokens (input + output across all recorded steps). */
  maxTotalTokens?: number;
  /** Hard ceiling on cumulative estimated USD cost, resolved via the pricing table. */
  maxTotalCostUsd?: number;
  /** Hard ceiling on wall-clock time since the guard was created. Enforced by an internal timer, independent of recordStep/check calls. */
  maxWallClockMs?: number;
  /** Hard ceiling on the number of recordStep() calls (i.e. loop iterations). */
  maxIterations?: number;
}

export interface ConvergenceConfig {
  /** Consecutive near-identical steps required before declaring convergence. @default 3 */
  minConsecutive?: number;
  /** Similarity score (0..1) above which two steps count as "the same". @default 0.95 */
  similarityThreshold?: number;
  /** Override the built-in similarity function. */
  similarityFn?: (a: string, b: string) => number;
}

export interface DriftConfig {
  /** Shortest oscillation period to test for (e.g. 2 = A,B,A,B). @default 2 */
  minCycleLength?: number;
  /** Longest oscillation period to test for. @default 3 */
  maxCycleLength?: number;
  /** Full cycle repeats required before declaring drift. @default 2 */
  minRepeats?: number;
  /** Average similarity score (0..1) across a cycle's pairs required to declare drift. @default 0.85 */
  similarityThreshold?: number;
  /** Per-pair minimum similarity; below this, the cycle is rejected even if the average clears the threshold. @default similarityThreshold - 0.15 */
  similarityFloor?: number;
  /** Override the built-in similarity function. */
  similarityFn?: (a: string, b: string) => number;
}

export interface LoopGuardConfig {
  budgets?: LoopGuardBudgets;
  /** `false` disables convergence detection entirely. Omit or `{}` for defaults. */
  convergence?: ConvergenceConfig | false;
  /** `false` disables drift detection entirely. Omit or `{}` for defaults. */
  drift?: DriftConfig | false;
  /** Merged over DEFAULT_PRICING; entries here win on key collision. */
  pricing?: PricingTable;
  /** Used to resolve provider/model for pricing when a given step doesn't specify its own. */
  defaultLLM?: LLMConfig;
  /** Ring-buffer size for content history. @default auto-derived from convergence/drift settings */
  historySize?: number;
  /** Cap on characters compared by the default similarity function (perf/safety bound). @default 2000 */
  maxCompareChars?: number;
  /** Fires exactly once, the first time the guard trips, for any reason. */
  onHalt?: (verdict: HaltVerdict) => void;
  /** Injectable clock, for tests. @default Date.now */
  now?: () => number;
}

export interface StepInput {
  /** Opaque text used for convergence/drift comparison. Guard treats this as a plain string — callers decide what's meaningful. */
  content?: string;
  /** Token usage for this step's LLM call, if known. */
  usage?: TokenUsage;
  /** Per-step override for pricing resolution; falls back to defaultLLM. */
  provider?: LLMProvider;
  model?: string;
  /** Set by the source loop when IT believes this is its terminal step. Strongest, cheapest convergence signal — never required. Only takes effect when convergence detection is enabled. */
  isFinal?: boolean;
}

export interface HaltVerdict {
  tripped: boolean;
  reason?: HaltReason;
  message?: string;
}

export interface GuardSnapshot {
  iterations: number;
  totalTokens: number;
  totalCostUsd: number;
  /** True if any recorded step had usage that couldn't be priced (missing pricing entry or missing token counts). */
  costUnknown: boolean;
  elapsedMs: number;
  tripped: boolean;
  reason?: HaltReason;
  message?: string;
}

export interface LoopGuardInstance {
  /** Aborts when the guard trips, for any reason. Thread this into corellm's `signal` on every LLM call. */
  readonly signal: AbortSignal;
  /** Record one iteration's outcome. Updates counters/history and evaluates all trip conditions immediately. */
  recordStep(step: StepInput): HaltVerdict;
  /** Re-evaluate trip conditions right now (re-checks wall-clock even with no new steps) and return the current verdict. */
  check(): HaltVerdict;
  /** Convenience boolean = check().tripped. */
  shouldHalt(): boolean;
  /** Point-in-time counters, safe to call anytime. */
  snapshot(): GuardSnapshot;
  /** Clears all counters/history and creates a fresh (non-aborted) internal AbortController — re-read `.signal` after calling this. */
  reset(): void;
}

/** Thrown as the AbortSignal's `.reason` when the guard trips; propagates through corellm's fetch layer into the caller's catch block. */
export class LoopGuardHaltError extends Error {
  readonly reason: HaltReason;
  constructor(reason: HaltReason, message?: string) {
    super(message ?? `Loop halted: ${reason}`);
    this.name = "LoopGuardHaltError";
    this.reason = reason;
  }
}
