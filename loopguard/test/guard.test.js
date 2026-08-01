const test = require("node:test");
const assert = require("node:assert/strict");
const { createLoopGuard } = require("../dist/index.js");

test("maxTotalTokens trips at the exact crossing point", () => {
  const guard = createLoopGuard({ budgets: { maxTotalTokens: 100 }, convergence: false, drift: false });
  const v1 = guard.recordStep({ usage: { inputTokens: 40, outputTokens: 30 } }); // total 70
  assert.equal(v1.tripped, false);
  const v2 = guard.recordStep({ usage: { inputTokens: 20, outputTokens: 20 } }); // total 110
  assert.equal(v2.tripped, true);
  assert.equal(v2.reason, "budget-tokens");
  assert.equal(guard.signal.aborted, true);
});

test("maxTotalCostUsd trips with the correct computed cost", () => {
  const pricing = { "anthropic:default": { inputPerMTokUsd: 10, outputPerMTokUsd: 20 } };
  const guard = createLoopGuard({
    budgets: { maxTotalCostUsd: 0.001 },
    pricing,
    defaultLLM: { provider: "anthropic", apiKey: "k" },
    convergence: false,
    drift: false,
  });
  const verdict = guard.recordStep({ usage: { inputTokens: 1000, outputTokens: 1000 } });
  assert.equal(verdict.tripped, true);
  assert.equal(verdict.reason, "budget-cost");
  const snap = guard.snapshot();
  assert.ok(Math.abs(snap.totalCostUsd - 0.03) < 1e-9, `expected ~0.03, got ${snap.totalCostUsd}`);
});

test("maxIterations trips after exactly N recordStep() calls", () => {
  const guard = createLoopGuard({ budgets: { maxIterations: 3 }, convergence: false, drift: false });
  assert.equal(guard.recordStep({}).tripped, false);
  assert.equal(guard.recordStep({}).tripped, false);
  const third = guard.recordStep({});
  assert.equal(third.tripped, true);
  assert.equal(third.reason, "budget-iterations");
  assert.equal(guard.snapshot().iterations, 3);
});

test("maxWallClockMs trips via an injectable fake clock with zero real elapsed time", () => {
  let fakeNow = 1000;
  const guard = createLoopGuard({
    now: () => fakeNow,
    budgets: { maxWallClockMs: 50 },
    convergence: false,
    drift: false,
  });
  assert.equal(guard.check().tripped, false);
  fakeNow = 1060; // 60ms of *fake* time, zero real time
  const verdict = guard.check();
  assert.equal(verdict.tripped, true);
  assert.equal(verdict.reason, "budget-time");
});

test("maxWallClockMs trips via a real internal timer with zero recordStep/check calls", async () => {
  const guard = createLoopGuard({ budgets: { maxWallClockMs: 20 }, convergence: false, drift: false });
  await new Promise((resolve) => setTimeout(resolve, 80));
  // No recordStep()/check() was ever called — the internal timer alone must have tripped it.
  assert.equal(guard.signal.aborted, true);
  const snap = guard.snapshot();
  assert.equal(snap.tripped, true);
  assert.equal(snap.reason, "budget-time");
});

test("onHalt fires exactly once across repeated post-trip calls", () => {
  const seen = [];
  const guard = createLoopGuard({
    budgets: { maxIterations: 1 },
    onHalt: (v) => seen.push(v),
    convergence: false,
    drift: false,
  });
  guard.recordStep({});
  guard.recordStep({});
  guard.check();
  guard.shouldHalt();
  assert.equal(seen.length, 1);
  assert.equal(seen[0].reason, "budget-iterations");
});

test("reset() clears counters and re-arms a fresh, non-aborted signal", () => {
  const guard = createLoopGuard({ budgets: { maxIterations: 1 }, convergence: false, drift: false });
  const signalBefore = guard.signal;
  guard.recordStep({});
  assert.equal(guard.snapshot().tripped, true);
  assert.equal(signalBefore.aborted, true);

  guard.reset();
  assert.equal(guard.snapshot().tripped, false);
  assert.equal(guard.snapshot().iterations, 0);
  const signalAfter = guard.signal;
  assert.notEqual(signalAfter, signalBefore);
  assert.equal(signalAfter.aborted, false);
});

test("trip precedence: iterations > tokens > cost when simultaneously true", () => {
  const pricing = { "anthropic:default": { inputPerMTokUsd: 10, outputPerMTokUsd: 20 } };
  const guard = createLoopGuard({
    budgets: { maxIterations: 1, maxTotalTokens: 10, maxTotalCostUsd: 0.0001 },
    pricing,
    defaultLLM: { provider: "anthropic", apiKey: "k" },
    convergence: false,
    drift: false,
  });
  // This single call simultaneously exceeds iterations, tokens, AND cost.
  const verdict = guard.recordStep({ usage: { inputTokens: 1000, outputTokens: 1000 } });
  assert.equal(verdict.tripped, true);
  assert.equal(verdict.reason, "budget-iterations");
});

test("trip precedence: time wins over iterations when both are true", () => {
  let fakeNow = 1000;
  const guard = createLoopGuard({
    now: () => fakeNow,
    budgets: { maxWallClockMs: 10, maxIterations: 1 },
    convergence: false,
    drift: false,
  });
  fakeNow = 1020; // exceeds the 10ms wall-clock budget before the first step is even recorded
  const verdict = guard.recordStep({});
  assert.equal(verdict.tripped, true);
  assert.equal(verdict.reason, "budget-time");
});

test("costUnknown is set when pricing can't resolve, without corrupting other budgets", () => {
  const guard = createLoopGuard({ budgets: { maxIterations: 5 }, convergence: false, drift: false });
  const verdict = guard.recordStep({ usage: { inputTokens: 100, outputTokens: 50 }, provider: "unknown-provider" });
  assert.equal(verdict.tripped, false);
  const snap = guard.snapshot();
  assert.equal(snap.costUnknown, true);
  assert.equal(snap.totalCostUsd, 0);
  assert.equal(snap.iterations, 1);
});

test("convergence: false and drift: false — repeated identical content never trips, only budgets can", () => {
  const guard = createLoopGuard({ convergence: false, drift: false });
  for (let i = 0; i < 20; i++) {
    const verdict = guard.recordStep({ content: "identical every time" });
    assert.equal(verdict.tripped, false);
  }
});

test("a config-less guard never trips on genuinely varying content", () => {
  // Note: strings sharing a long common template (e.g. "step N of task") stay
  // similar under edit-distance regardless of N — these fixtures are
  // deliberately lexically unrelated to each other to avoid that trap.
  const distinctPhrases = [
    "quantum flux stabilizer readings", "banana harvest report for march",
    "orbital decay calculation complete", "sandwich recipe variant fourteen",
    "glacier melt rate survey data", "violin string tension adjustment",
    "asteroid mining feasibility study", "coffee bean roast profile log",
    "submarine ballast tank pressure", "lighthouse beam rotation schedule",
    "volcano seismic activity index", "beekeeping colony health check",
    "satellite uplink frequency scan", "cheese aging cellar humidity",
    "wind turbine blade inspection", "coral reef bleaching assessment",
    "vineyard soil acidity test", "telescope mirror alignment check",
    "waterfall flow rate measurement", "meteor shower visibility forecast",
  ];
  const guard = createLoopGuard();
  for (const content of distinctPhrases) {
    const verdict = guard.recordStep({ content });
    assert.equal(verdict.tripped, false);
  }
});

test("convergence trips end-to-end through recordStep on repeated similar content", () => {
  const guard = createLoopGuard({ drift: false });
  assert.equal(guard.recordStep({ content: "the final answer is 42" }).tripped, false);
  assert.equal(guard.recordStep({ content: "the final answer is 42" }).tripped, false);
  const third = guard.recordStep({ content: "the final answer is 42" });
  assert.equal(third.tripped, true);
  assert.equal(third.reason, "converged");
});

test("isFinal is an immediate convergence signal when convergence is enabled", () => {
  const guard = createLoopGuard({ drift: false });
  const verdict = guard.recordStep({ content: "answer: 7", isFinal: true });
  assert.equal(verdict.tripped, true);
  assert.equal(verdict.reason, "converged");
});

test("isFinal has no effect when convergence detection is disabled", () => {
  const guard = createLoopGuard({ convergence: false, drift: false });
  const verdict = guard.recordStep({ content: "answer: 7", isFinal: true });
  assert.equal(verdict.tripped, false);
});

test("drift trips end-to-end through recordStep on an alternating tool-call pattern", () => {
  const guard = createLoopGuard({ convergence: false });
  const seq = ["call:search(a)", "call:search(b)"];
  let last;
  for (let i = 0; i < 6; i++) {
    last = guard.recordStep({ content: seq[i % 2] });
  }
  assert.equal(last.tripped, true);
  assert.equal(last.reason, "drift");
});

test("with default settings, convergence fires before drift ever gets enough history for identical repeats", () => {
  // Convergence needs only minConsecutive (3) identical steps; drift's
  // period-2 default needs period*(minRepeats+1) = 6. So for a simple
  // identical-repeat run, convergence always wins by construction, not by
  // an explicit precedence rule — the guard trips well before drift's
  // window is ever full enough to evaluate.
  const guard = createLoopGuard({});
  assert.equal(guard.recordStep({ content: "identical every time" }).tripped, false);
  assert.equal(guard.recordStep({ content: "identical every time" }).tripped, false);
  const third = guard.recordStep({ content: "identical every time" });
  assert.equal(third.tripped, true);
  assert.equal(third.reason, "converged");
});
