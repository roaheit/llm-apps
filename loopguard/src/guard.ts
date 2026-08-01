import { DEFAULT_PRICING, estimateCostUsd, resolvePricing } from "./pricing";
import { detectConvergence, type ResolvedConvergenceConfig } from "./convergence";
import { detectDrift, type ResolvedDriftConfig } from "./drift";
import { normalizedSimilarity } from "./similarity";
import {
  LoopGuardHaltError,
  type LoopGuardConfig,
  type LoopGuardInstance,
  type StepInput,
  type HaltVerdict,
  type GuardSnapshot,
  type HaltReason,
} from "./types";

const DEFAULT_MAX_COMPARE_CHARS = 2000;
const DEFAULT_HISTORY_FLOOR = 6;

export function createLoopGuard(config: LoopGuardConfig = {}): LoopGuardInstance {
  const now = config.now ?? Date.now;
  const budgets = config.budgets ?? {};
  const maxCompareChars = config.maxCompareChars ?? DEFAULT_MAX_COMPARE_CHARS;
  const defaultSimilarityFn = (a: string, b: string) => normalizedSimilarity(a, b, maxCompareChars);

  let resolvedConvergence: ResolvedConvergenceConfig | undefined;
  if (config.convergence !== false) {
    const cfg = config.convergence ?? {};
    resolvedConvergence = {
      minConsecutive: cfg.minConsecutive ?? 3,
      similarityThreshold: cfg.similarityThreshold ?? 0.95,
      similarityFn: cfg.similarityFn ?? defaultSimilarityFn,
    };
  }

  let resolvedDrift: ResolvedDriftConfig | undefined;
  if (config.drift !== false) {
    const cfg = config.drift ?? {};
    const similarityThreshold = cfg.similarityThreshold ?? 0.85;
    resolvedDrift = {
      minCycleLength: cfg.minCycleLength ?? 2,
      maxCycleLength: cfg.maxCycleLength ?? 3,
      minRepeats: cfg.minRepeats ?? 2,
      similarityThreshold,
      similarityFloor: cfg.similarityFloor ?? similarityThreshold - 0.15,
      similarityFn: cfg.similarityFn ?? defaultSimilarityFn,
    };
  }

  // Auto-derived so raising drift.maxCycleLength can never silently starve
  // itself of enough history to ever fire.
  const historySize =
    config.historySize ??
    Math.max(
      resolvedConvergence?.minConsecutive ?? 0,
      resolvedDrift ? resolvedDrift.maxCycleLength * (resolvedDrift.minRepeats + 1) : 0,
      DEFAULT_HISTORY_FLOOR
    );

  const pricingTable = { ...DEFAULT_PRICING, ...(config.pricing ?? {}) };

  if (budgets.maxTotalCostUsd != null) {
    const provider = config.defaultLLM?.provider;
    const model = config.defaultLLM?.model;
    const resolved = provider ? resolvePricing(pricingTable, provider, model) : undefined;
    if (!resolved) {
      console.warn(
        "loopguard: budgets.maxTotalCostUsd is set but no pricing entry can be resolved" +
          (provider
            ? ` for provider "${provider}"${model ? ` model "${model}"` : ""}.`
            : " (no config.defaultLLM.provider set).") +
          " This cost ceiling will never trip. Pass `pricing` with a matching entry, or set config.defaultLLM."
      );
    }
  }

  const startedAt = now();
  let iterations = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalCostUsd = 0;
  let costUnknown = false;
  let history: string[] = [];
  let controller = new AbortController();
  let wallClockTimer: ReturnType<typeof setTimeout> | undefined;
  let verdict: HaltVerdict = { tripped: false };

  function trip(reason: HaltReason, message: string): void {
    if (verdict.tripped) return;
    verdict = { tripped: true, reason, message };
    if (wallClockTimer) clearTimeout(wallClockTimer);
    controller.abort(new LoopGuardHaltError(reason, message));
    config.onHalt?.(verdict);
  }

  function armWallClockTimer(): void {
    if (wallClockTimer) clearTimeout(wallClockTimer);
    if (budgets.maxWallClockMs == null) return;
    wallClockTimer = setTimeout(() => {
      trip("budget-time", `Exceeded max wall-clock time: ${budgets.maxWallClockMs}ms`);
    }, budgets.maxWallClockMs);
    // Don't let this timer alone keep a Node process alive (no-op in browsers).
    (wallClockTimer as unknown as { unref?: () => void }).unref?.();
  }

  function evaluateBudgets(): HaltVerdict | undefined {
    const elapsedMs = now() - startedAt;
    if (budgets.maxWallClockMs != null && elapsedMs >= budgets.maxWallClockMs) {
      return {
        tripped: true,
        reason: "budget-time",
        message: `Exceeded max wall-clock time: ${elapsedMs}ms / ${budgets.maxWallClockMs}ms`,
      };
    }
    if (budgets.maxIterations != null && iterations >= budgets.maxIterations) {
      return {
        tripped: true,
        reason: "budget-iterations",
        message: `Exceeded max iterations: ${iterations} / ${budgets.maxIterations}`,
      };
    }
    const totalTokens = totalInputTokens + totalOutputTokens;
    if (budgets.maxTotalTokens != null && totalTokens >= budgets.maxTotalTokens) {
      return {
        tripped: true,
        reason: "budget-tokens",
        message: `Exceeded max tokens: ${totalTokens} / ${budgets.maxTotalTokens}`,
      };
    }
    if (budgets.maxTotalCostUsd != null && totalCostUsd >= budgets.maxTotalCostUsd) {
      return {
        tripped: true,
        reason: "budget-cost",
        message: `Exceeded max cost: $${totalCostUsd.toFixed(4)} / $${budgets.maxTotalCostUsd}`,
      };
    }
    return undefined;
  }

  function checkNow(): HaltVerdict {
    if (verdict.tripped) return verdict;
    const halt = evaluateBudgets();
    if (halt) trip(halt.reason as HaltReason, halt.message as string);
    return verdict;
  }

  armWallClockTimer();

  return {
    get signal() {
      return controller.signal;
    },

    recordStep(step: StepInput): HaltVerdict {
      if (verdict.tripped) return verdict;

      iterations += 1;
      if (step.usage) {
        totalInputTokens += step.usage.inputTokens ?? 0;
        totalOutputTokens += step.usage.outputTokens ?? 0;
        const provider = step.provider ?? config.defaultLLM?.provider;
        const model = step.model ?? config.defaultLLM?.model;
        const pricing = provider ? resolvePricing(pricingTable, provider, model) : undefined;
        const cost = estimateCostUsd(step.usage, pricing);
        if (cost) totalCostUsd += cost.usd;
        else costUnknown = true;
      }

      if (step.content != null) {
        history.push(step.content);
        if (history.length > historySize * 2) history = history.slice(-historySize);
      }

      const budgetHalt = evaluateBudgets();
      if (budgetHalt) {
        trip(budgetHalt.reason as HaltReason, budgetHalt.message as string);
        return verdict;
      }

      // An explicit "I'm done" signal from the source loop is authoritative and
      // is checked before the heuristic drift scan below — an incidental cycle
      // match on the loop's own genuine terminal output should never
      // masquerade as "stuck oscillating" when the loop told us directly that
      // it's finished.
      if (resolvedConvergence && step.isFinal) {
        trip("converged", "Source loop signaled isFinal");
        return verdict;
      }

      const recentHistory = history.slice(-historySize);

      if (resolvedDrift) {
        const { drift, period } = detectDrift(recentHistory, resolvedDrift);
        if (drift) {
          trip("drift", `Detected oscillation with cycle length ${period}`);
          return verdict;
        }
      }

      if (resolvedConvergence) {
        const { converged } = detectConvergence(recentHistory, resolvedConvergence);
        if (converged) {
          trip(
            "converged",
            `${resolvedConvergence.minConsecutive} consecutive steps exceeded similarity threshold ${resolvedConvergence.similarityThreshold}`
          );
          return verdict;
        }
      }

      return verdict;
    },

    check(): HaltVerdict {
      return checkNow();
    },

    shouldHalt(): boolean {
      return checkNow().tripped;
    },

    snapshot(): GuardSnapshot {
      return {
        iterations,
        totalTokens: totalInputTokens + totalOutputTokens,
        totalCostUsd,
        costUnknown,
        elapsedMs: now() - startedAt,
        tripped: verdict.tripped,
        reason: verdict.reason,
        message: verdict.message,
      };
    },

    reset(): void {
      iterations = 0;
      totalInputTokens = 0;
      totalOutputTokens = 0;
      totalCostUsd = 0;
      costUnknown = false;
      history = [];
      verdict = { tripped: false };
      controller = new AbortController();
      armWallClockTimer();
    },
  };
}
