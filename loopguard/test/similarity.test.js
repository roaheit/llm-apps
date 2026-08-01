const test = require("node:test");
const assert = require("node:assert/strict");
const { normalizedSimilarity } = require("../dist/index.js");

test("exact match after normalization returns 1", () => {
  assert.equal(normalizedSimilarity("Hello World", "hello world"), 1);
});

test("whitespace/case-only differences return 1", () => {
  assert.equal(normalizedSimilarity("  Search(query)  ", "search(query)"), 1);
  assert.equal(normalizedSimilarity("a   b\tc", "a b c"), 1);
});

test("substantially different strings return a low score", () => {
  const score = normalizedSimilarity("abcdef", "xyz123");
  assert.ok(score < 0.3, `expected a low score, got ${score}`);
});

test("empty vs non-empty string returns 0, both empty returns 1", () => {
  assert.equal(normalizedSimilarity("", ""), 1);
  assert.equal(normalizedSimilarity("", "something"), 0);
});

test("input longer than maxCompareChars stays bounded (no crash/NaN)", () => {
  const a = "x".repeat(5000);
  const b = "y".repeat(5000);
  const score = normalizedSimilarity(a, b, 2000);
  assert.ok(Number.isFinite(score), `expected a finite score, got ${score}`);
  assert.ok(score >= 0 && score <= 1, `expected score in [0,1], got ${score}`);

  const identical = normalizedSimilarity("z".repeat(5000), "z".repeat(5000), 2000);
  assert.equal(identical, 1);
});
