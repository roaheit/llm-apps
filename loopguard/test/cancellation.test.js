const test = require("node:test");
const assert = require("node:assert/strict");
const { createLoopGuard, LoopGuardHaltError } = require("../dist/index.js");
const { complete } = require("corellm");

test("a wall-clock trip cancels an in-flight corellm call, not just future ones", async () => {
  const realFetch = globalThis.fetch;
  let sawAbort = false;

  globalThis.fetch = (_url, init) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => resolve(new Response(JSON.stringify({ content: [{ type: "text", text: "too slow" }] }), { status: 200 })),
        2000
      );
      init.signal?.addEventListener(
        "abort",
        () => {
          sawAbort = true;
          clearTimeout(timer);
          reject(init.signal.reason ?? new Error("aborted"));
        },
        { once: true }
      );
    });

  try {
    const guard = createLoopGuard({ budgets: { maxWallClockMs: 20 }, convergence: false, drift: false });
    const start = Date.now();

    await assert.rejects(
      () => complete({ provider: "anthropic", apiKey: "k" }, { prompt: "p", signal: guard.signal }),
      (err) => err instanceof LoopGuardHaltError && err.reason === "budget-time"
    );

    const elapsed = Date.now() - start;
    assert.ok(elapsed < 500, `expected the call to abort quickly, took ${elapsed}ms`);
    assert.equal(sawAbort, true);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("an externally-aborted guard signal cancels an in-flight corellm call", async () => {
  const realFetch = globalThis.fetch;
  let sawAbort = false;

  globalThis.fetch = (_url, init) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(() => resolve(new Response(JSON.stringify({ content: [] }), { status: 200 })), 2000);
      init.signal?.addEventListener(
        "abort",
        () => {
          sawAbort = true;
          clearTimeout(timer);
          reject(init.signal.reason ?? new Error("aborted"));
        },
        { once: true }
      );
    });

  try {
    const guard = createLoopGuard({ budgets: { maxIterations: 1 }, convergence: false, drift: false });
    const callPromise = complete({ provider: "anthropic", apiKey: "k" }, { prompt: "p", signal: guard.signal });

    // Simulate the loop's next iteration deciding to halt while the call above is still in flight.
    setTimeout(() => guard.recordStep({}), 10);

    await assert.rejects(() => callPromise, (err) => err instanceof LoopGuardHaltError && err.reason === "budget-iterations");
    assert.equal(sawAbort, true);
  } finally {
    globalThis.fetch = realFetch;
  }
});
