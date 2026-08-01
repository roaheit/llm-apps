# loopguard

Budget ceilings, convergence &amp; drift detection, and real cancellation for
**any** agentic loop. An external observer you call into — no framework to
adopt, no control flow to hand over.

Part of [llm-apps](https://github.com/roaheit/llm-apps) — a growing collection
of real-world LLM applications &amp; AI agents.

- **Budgets** — hard ceilings on total tokens, estimated USD cost, wall-clock
  time, and iteration count. Any dimension left unset is simply not enforced.
- **Convergence detection** — is the loop stuck repeating one answer? Several
  consecutive near-identical outputs (or an explicit `isFinal` hint) halt with
  reason `"converged"`.
- **Drift detection** — is the loop oscillating between ≥2 states
  (`A → B → A → B`)? An averaged-similarity scan with a per-pair floor catches
  it, with reason `"drift"`.
- **Real cancellation** — `guard.signal` is a genuine `AbortSignal`. Thread it
  into [corellm](../corellm)'s `signal` and a trip aborts an **in-flight** LLM
  call, not just future ones. Wall-clock budgets are enforced by an internal
  timer, independent of whether your loop ever calls back in.
- **Zero extra cost** — convergence/drift use plain string similarity
  (normalized Levenshtein distance) by default. No embeddings, no extra
  network calls, no added latency per iteration.
- **Framework-agnostic** — no React, no Node-only APIs, no dependency on any
  other package in this repo except `corellm`'s types.

---

## Install

```bash
npm install loopguard
```

## Quick start (any loop)

```ts
import { createLoopGuard } from "loopguard";
import { complete } from "corellm";

const guard = createLoopGuard({
  budgets: { maxTotalTokens: 50_000, maxTotalCostUsd: 2, maxIterations: 25 },
});

while (!guard.shouldHalt()) {
  const result = await complete(llm, { prompt, signal: guard.signal });
  const verdict = guard.recordStep({ content: result.text, usage: result.usage });
  if (verdict.tripped) break; // verdict.reason: "budget-tokens" | "budget-cost" | ... | "converged" | "drift"
}

console.log(guard.snapshot()); // { iterations, totalTokens, totalCostUsd, elapsedMs, ... }
```

This example has no dependency on any other `llm-apps` package — `loopguard`
works with any loop that can call `recordStep()`/`shouldHalt()`, in Node, the
browser, or anywhere `fetch` and `AbortController` exist.

## Retrofitting an existing loop

Wiring `loopguard` into an existing loop is a handful of small, additive
changes — no restructuring required. This is the exact shape used to retrofit
[tool-pilot](../tool-pilot)'s ReAct loop:

```ts
const guard = config.guard ? createLoopGuard({ ...config.guard, defaultLLM: config.llm }) : undefined;

for (let i = 0; i < maxSteps; i++) {
  // 1. Check at the top of every iteration.
  if (guard?.shouldHalt()) return buildHaltedRun(guard.check().reason);

  // 2. Thread the signal into your LLM call for real cancellation.
  const result = await complete(config.llm, { prompt, system, signal: guard?.signal });

  // 3. Record the outcome — usage feeds budgets, content feeds convergence/drift.
  guard?.recordStep({ content: actionText, usage: result.usage, isFinal: isDone });

  // ...your existing loop logic, completely untouched.
}
```

`guard` is `undefined` when the caller doesn't opt in, and every method call
above is guarded with `?.` — so this integration is a pure opt-in: callers who
never set a `guard` config see zero behavior change.

## How convergence &amp; drift detection work

Both run over a small rolling window of the `content` strings you pass to
`recordStep()`, using a zero-cost string-similarity check (normalized
Levenshtein distance) — not embeddings.

- **Convergence** looks at the last few steps: if they're all highly similar
  to each other, the loop has stopped making progress and is declared
  `"converged"`. Passing `isFinal: true` on a step (e.g. your loop's own
  "final answer" signal) is a free, immediate convergence signal.
- **Drift** looks for a repeating cycle — comparing each step to the one
  `N` positions back, for a few candidate cycle lengths. It uses the
  **average** similarity across a cycle (with a per-pair floor) rather than
  requiring every position to match exactly, so a loop that repeats the same
  tool call while rewording its reasoning each time still gets caught.

Both are opt-out (`convergence: false` / `drift: false`) and fully
configurable — thresholds, cycle lengths, and even the similarity function
itself (`similarityFn`) can be overridden.

## Config reference

| Field | Type | Description |
|---|---|---|
| `budgets.maxTotalTokens` | `number` | Cumulative input + output tokens across all recorded steps. |
| `budgets.maxTotalCostUsd` | `number` | Estimated USD cost via the `pricing` table. Warns if it can never resolve. |
| `budgets.maxWallClockMs` | `number` | Wall-clock ceiling, enforced by an internal timer — trips even with zero `recordStep()` calls. |
| `budgets.maxIterations` | `number` | Hard cap on the number of `recordStep()` calls. |
| `convergence` | `ConvergenceConfig \| false` | Tune `minConsecutive`, `similarityThreshold`, or supply a custom `similarityFn`. `false` disables it. |
| `drift` | `DriftConfig \| false` | Tune `minCycleLength`, `maxCycleLength`, `minRepeats`, `similarityThreshold`/`similarityFloor`, or `similarityFn`. `false` disables it. |
| `pricing` | `PricingTable` | Merged over `DEFAULT_PRICING` (illustrative — verify against real provider pricing). |
| `defaultLLM` | `LLMConfig` | Resolves provider/model for pricing when a step doesn't specify its own. |
| `onHalt` | `(verdict) => void` | Fires exactly once, the first time the guard trips, for any reason. |

### `LoopGuardInstance`

| Member | Description |
|---|---|
| `signal: AbortSignal` | Aborts when the guard trips. Thread into corellm's `signal` on every LLM call. |
| `recordStep(step)` | Record one iteration's `content`/`usage`. Evaluates every trip condition immediately. |
| `check()` | Re-evaluate right now (re-checks wall-clock even with no new steps). |
| `shouldHalt()` | Convenience `boolean` — `check().tripped`. |
| `snapshot()` | Point-in-time counters: iterations, tokens, cost, elapsed time, trip state. |
| `reset()` | Clears all counters and re-arms a fresh, non-aborted signal. |

### Trip precedence

When several conditions are true on the same call, the reported `reason`
follows: **wall-clock time → iterations → tokens → cost → drift → converged**
— hard budgets always take priority over the softer heuristic signals, and an
explicit `isFinal` hint always wins over drift (an authoritative "I'm done"
should never be overridden by an incidental cycle match).

## Known limitations

- Only the wall-clock budget is truly preemptive — the others can only ever
  gate the *next* iteration, since their data isn't known until the current
  step's call returns.
- `recordStep()`'s `content` is treated as an opaque string; convergence and
  drift quality depend entirely on what you pass. Passing nothing disables
  both silently (you still get budget enforcement).
- The bundled `DEFAULT_PRICING` table is illustrative and will drift from
  real provider prices over time — override it with `pricing` for anything
  cost-sensitive.

---

## License

MIT
