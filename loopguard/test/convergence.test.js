const test = require("node:test");
const assert = require("node:assert/strict");
const { detectConvergence, normalizedSimilarity } = require("../dist/index.js");

const defaultCfg = {
  minConsecutive: 3,
  similarityThreshold: 0.95,
  similarityFn: normalizedSimilarity,
};

test("3 identical strings converge", () => {
  const { converged } = detectConvergence(["same answer", "same answer", "same answer"], defaultCfg);
  assert.equal(converged, true);
});

test("3 meaningfully different strings do not converge", () => {
  const { converged } = detectConvergence(["alpha result", "beta result differs", "gamma outcome entirely"], defaultCfg);
  assert.equal(converged, false);
});

test("fewer than minConsecutive entries never converge (insufficient data, not a false positive)", () => {
  const { converged } = detectConvergence(["same", "same"], defaultCfg);
  assert.equal(converged, false);
});

test("custom threshold/minConsecutive change the outcome", () => {
  const history = ["the cat sat", "the cat sit", "the cat sad"]; // similar but not identical
  const strict = detectConvergence(history, { ...defaultCfg, similarityThreshold: 0.99 });
  const lenient = detectConvergence(history, { ...defaultCfg, similarityThreshold: 0.5 });
  assert.equal(strict.converged, false);
  assert.equal(lenient.converged, true);

  const needsFour = detectConvergence(["x", "x", "x"], { ...defaultCfg, minConsecutive: 4 });
  assert.equal(needsFour.converged, false);
});
