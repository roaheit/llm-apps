const test = require("node:test");
const assert = require("node:assert/strict");
const { runAgent } = require("../dist/index.js");

function mockJsonResponse(text, usage) {
  return new Response(JSON.stringify({ content: [{ type: "text", text }], ...(usage ? { usage } : {}) }), {
    status: 200,
  });
}

test("regression: runAgent with no guard behaves as before across a multi-step run", async () => {
  const realFetch = globalThis.fetch;
  let call = 0;
  globalThis.fetch = async () => {
    call++;
    const text =
      call === 1
        ? 'THINK: I should look this up.\nACTION: lookup\nACTION_INPUT: {"term":"answer"}'
        : "THINK: Now I know.\nANSWER: 42";
    return mockJsonResponse(text);
  };
  try {
    const lookupTool = {
      name: "lookup",
      description: "looks things up",
      parameters: [{ name: "term", type: "string", description: "term", required: true }],
      execute: async (args) => `found: ${args.term}`,
    };
    const onStepKinds = [];
    const config = { llm: { provider: "anthropic", apiKey: "k" }, tools: [lookupTool] };
    const run = await runAgent("What is the answer?", config, (s) => onStepKinds.push(s.kind));

    assert.equal(run.status, "done");
    assert.equal(run.answer, "42");
    assert.equal(run.haltReason, undefined);
    assert.equal(call, 2);
    assert.deepEqual(onStepKinds, ["thinking", "tool-call", "observation", "thinking", "answer"]);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("LLM call throwing results in status: error (unchanged by the retrofit)", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 });
  try {
    const config = { llm: { provider: "anthropic", apiKey: "bad" }, tools: [] };
    const run = await runAgent("anything", config);
    assert.equal(run.status, "error");
    assert.ok(run.error);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("max steps exceeded forces an answer with status: done (unchanged by the retrofit)", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => mockJsonResponse('THINK: keep going\nACTION: search\nACTION_INPUT: {}');
  try {
    const config = { llm: { provider: "anthropic", apiKey: "k" }, tools: [], maxSteps: 2 };
    const run = await runAgent("never finishes", config);
    assert.equal(run.status, "done");
    assert.match(run.answer, /reached the maximum of 2 reasoning steps/);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("no THINK/ACTION/ANSWER markers falls back to the raw output as the answer (unchanged)", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => mockJsonResponse("Just a plain response with no format markers.");
  try {
    const config = { llm: { provider: "anthropic", apiKey: "k" }, tools: [] };
    const run = await runAgent("say something", config);
    assert.equal(run.status, "done");
    assert.equal(run.answer, "Just a plain response with no format markers.");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("guard.budgets.maxIterations halts a runaway tool-call loop after exactly N LLM calls", async () => {
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return mockJsonResponse('THINK: still working\nACTION: search\nACTION_INPUT: {"q":"x"}');
  };
  try {
    const config = {
      llm: { provider: "anthropic", apiKey: "k" },
      tools: [],
      guard: { budgets: { maxIterations: 2 }, convergence: false, drift: false },
    };
    const run = await runAgent("loop forever", config);
    // The trip is detected on the 2nd recorded step but only acted on at the
    // top of the *next* iteration — so exactly 2 LLM calls happen, not 3.
    assert.equal(calls, 2);
    assert.equal(run.status, "halted");
    assert.equal(run.haltReason, "budget-iterations");
    assert.ok(run.steps.some((s) => s.kind === "halted"));
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("guard.budgets.maxTotalTokens halts at the mathematically correct cumulative count", async () => {
  const realFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    return mockJsonResponse('THINK: still working\nACTION: search\nACTION_INPUT: {}', {
      input_tokens: 100,
      output_tokens: 50,
    });
  };
  try {
    const config = {
      llm: { provider: "anthropic", apiKey: "k" },
      tools: [],
      guard: { budgets: { maxTotalTokens: 300 }, convergence: false, drift: false },
    };
    const run = await runAgent("loop forever", config);
    // 150 tokens/call: call 1 -> 150 (not tripped), call 2 -> 300 (tripped).
    // This also indirectly proves the callLLM -> complete swap is wired
    // correctly — a regression there would silently make usage always
    // undefined, and this budget would never trip at all.
    assert.equal(calls, 2);
    assert.equal(run.status, "halted");
    assert.equal(run.haltReason, "budget-tokens");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("guard.drift halts on an alternating tool-call pattern using default drift settings", async () => {
  const realFetch = globalThis.fetch;
  let call = 0;
  const responses = [
    'THINK: trying search\nACTION: search\nACTION_INPUT: {"q":"a"}',
    'THINK: trying browse\nACTION: browse\nACTION_INPUT: {"url":"b"}',
  ];
  globalThis.fetch = async () => {
    const text = responses[call % 2];
    call++;
    return mockJsonResponse(text);
  };
  try {
    const config = {
      llm: { provider: "anthropic", apiKey: "k" },
      tools: [],
      guard: { convergence: false }, // drift stays at its defaults (period 2, needs 6 steps)
    };
    const run = await runAgent("loop forever", config);
    assert.equal(call, 6);
    assert.equal(run.status, "halted");
    assert.equal(run.haltReason, "drift");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("guard.budgets.maxWallClockMs cancels a hanging LLM call end-to-end", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = (_url, init) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(() => resolve(mockJsonResponse("THINK: x\nANSWER: too slow")), 2000);
      init.signal?.addEventListener(
        "abort",
        () => {
          clearTimeout(timer);
          reject(init.signal.reason ?? new Error("aborted"));
        },
        { once: true }
      );
    });
  try {
    const start = Date.now();
    const config = {
      llm: { provider: "anthropic", apiKey: "k" },
      tools: [],
      guard: { budgets: { maxWallClockMs: 30 }, convergence: false, drift: false },
    };
    const run = await runAgent("hang please", config);
    const elapsed = Date.now() - start;

    assert.ok(elapsed < 1000, `expected runAgent to resolve quickly, took ${elapsed}ms`);
    assert.equal(run.status, "halted");
    assert.equal(run.haltReason, "budget-time");
  } finally {
    globalThis.fetch = realFetch;
  }
});
