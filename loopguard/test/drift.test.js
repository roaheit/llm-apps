const test = require("node:test");
const assert = require("node:assert/strict");
const { detectDrift } = require("../dist/index.js");

const exactMatchFn = (a, b) => (a === b ? 1 : 0);

const baseCfg = {
  minCycleLength: 2,
  maxCycleLength: 3,
  minRepeats: 2,
  similarityThreshold: 0.85,
  similarityFloor: 0.7,
  similarityFn: exactMatchFn,
};

test("A,B,A,B,A,B is detected as period-2 drift", () => {
  const { drift, period } = detectDrift(["A", "B", "A", "B", "A", "B"], baseCfg);
  assert.equal(drift, true);
  assert.equal(period, 2);
});

test("A,B,C,A,B,C,A,B,C is detected as period-3 drift", () => {
  const history = ["A", "B", "C", "A", "B", "C", "A", "B", "C"];
  const { drift, period } = detectDrift(history, baseCfg);
  assert.equal(drift, true);
  assert.equal(period, 3);
});

test("a monotonically-changing sequence is not drift", () => {
  const history = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];
  const { drift } = detectDrift(history, baseCfg);
  assert.equal(drift, false);
});

test("history shorter than the required window is not drift", () => {
  const { drift } = detectDrift(["A", "B", "A"], baseCfg);
  assert.equal(drift, false);
});

test("average-with-floor tolerates partial novelty in one cycle position", () => {
  // Simulates a real ReAct "stuck" pattern: the tool-call half of the cycle
  // repeats verbatim, but the "thinking" half is legitimately reworded each
  // lap. A strict "every position must match" check (threshold as the floor)
  // would miss this; average-with-floor (floor below threshold) should not.
  const call = "search(query)";
  const partialMatchFn = (a, b) => {
    if (a === b) return 1;
    if (a.startsWith("think") && b.startsWith("think")) return 0.8; // reworded, but related
    return 0;
  };
  const history = [call, "think-v1", call, "think-v2", call, "think-v3"];
  const cfg = { ...baseCfg, similarityFn: partialMatchFn };

  const { drift, period } = detectDrift(history, cfg);
  assert.equal(drift, true);
  assert.equal(period, 2);

  // Prove this specifically depends on the floor being below the threshold:
  // with a strict floor equal to the threshold, the 0.8 "thinking" score
  // would be rejected outright.
  const strict = detectDrift(history, { ...cfg, similarityFloor: cfg.similarityThreshold });
  assert.equal(strict.drift, false);
});
