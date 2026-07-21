const test = require("node:test");
const assert = require("node:assert/strict");
const { complete, stream, extractJson, LLMError, DEFAULT_MODELS } = require("../dist/index.js");

const realFetch = globalThis.fetch;
const restore = () => { globalThis.fetch = realFetch; };
const sse = (frames) =>
  new Response(frames.map((f) => (typeof f === "string" ? f : `data: ${JSON.stringify(f)}`)).join("\n\n") + "\n\n", { status: 200 });

test("extractJson: fenced / preamble / nested / array / string-braces / failure", () => {
  assert.deepEqual(extractJson('{"a":1}'), { a: 1 });
  assert.deepEqual(extractJson("```json\n{\"a\":1}\n```"), { a: 1 });
  assert.deepEqual(extractJson('Sure:\n{"a":1,"b":[1,2]}\nthanks'), { a: 1, b: [1, 2] });
  assert.deepEqual(extractJson('{"filter":{"nested":true},"q":"x"}'), { filter: { nested: true }, q: "x" });
  assert.deepEqual(extractJson("[1,2,3]"), [1, 2, 3]);
  assert.deepEqual(extractJson('{"txt":"a } b { c"}'), { txt: "a } b { c" });
  assert.throws(() => extractJson("no json here"), LLMError);
});

test("complete: anthropic request shape + usage mapping", async () => {
  let body;
  globalThis.fetch = async (_u, init) => {
    body = JSON.parse(init.body);
    return new Response(JSON.stringify({ model: body.model, content: [{ type: "text", text: "Hi" }], stop_reason: "end_turn", usage: { input_tokens: 4, output_tokens: 2 } }), { status: 200 });
  };
  const r = await complete({ provider: "anthropic", apiKey: "k" }, { prompt: "p", system: "s" });
  assert.equal(body.model, DEFAULT_MODELS.anthropic);
  assert.equal(body.system, "s");
  assert.equal(r.text, "Hi");
  assert.deepEqual(r.usage, { inputTokens: 4, outputTokens: 2, totalTokens: 6 });
  restore();
});

test("complete: retries on 429 then succeeds", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls++;
    if (calls === 1) return new Response("rate", { status: 429, headers: { "retry-after": "0" } });
    return new Response(JSON.stringify({ content: [{ type: "text", text: "ok" }] }), { status: 200 });
  };
  const r = await complete({ provider: "anthropic", apiKey: "k", maxRetries: 2 }, { prompt: "p" });
  assert.equal(calls, 2);
  assert.equal(r.text, "ok");
  restore();
});

test("complete: surfaces non-retryable error (status + message, no retry)", async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; return new Response(JSON.stringify({ error: { message: "bad key" } }), { status: 401 }); };
  await assert.rejects(
    () => complete({ provider: "anthropic", apiKey: "k", maxRetries: 3 }, { prompt: "p" }),
    (e) => e instanceof LLMError && e.status === 401 && /bad key/.test(e.message)
  );
  assert.equal(calls, 1);
  restore();
});

test("complete: custom provider requires adapter or baseUrl", async () => {
  await assert.rejects(() => complete({ provider: "custom" }, { prompt: "p" }), LLMError);
});

test("complete: openai omits empty system message + maps usage + responseFormat", async () => {
  let body;
  globalThis.fetch = async (_u, init) => {
    body = JSON.parse(init.body);
    return new Response(JSON.stringify({ model: "gpt-4o", choices: [{ message: { content: "oai" }, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } }), { status: 200 });
  };
  const r = await complete({ provider: "openai", apiKey: "k" }, { prompt: "p", responseFormat: "json" });
  assert.equal(body.messages.length, 1);
  assert.equal(body.messages[0].role, "user");
  assert.deepEqual(body.response_format, { type: "json_object" });
  assert.equal(r.text, "oai");
  assert.equal(r.usage.totalTokens, 5);
  restore();
});

test("stream: anthropic deltas + usage + finishReason", async () => {
  globalThis.fetch = async () => sse([
    { type: "message_start", message: { model: "claude-sonnet-5", usage: { input_tokens: 10 } } },
    { type: "content_block_delta", delta: { type: "text_delta", text: "Hel" } },
    { type: "content_block_delta", delta: { type: "text_delta", text: "lo!" } },
    { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
    { type: "message_stop" },
  ]);
  const toks = [];
  const r = await stream({ provider: "anthropic", apiKey: "k" }, { prompt: "p", onToken: (d) => toks.push(d) });
  assert.equal(toks.join(""), "Hello!");
  assert.equal(r.text, "Hello!");
  assert.equal(r.finishReason, "end_turn");
  assert.deepEqual(r.usage, { inputTokens: 10, outputTokens: 5, totalTokens: 15 });
  restore();
});

test("stream: chat deltas + final-chunk usage", async () => {
  globalThis.fetch = async () => sse([
    { choices: [{ delta: { content: "A" } }] },
    { choices: [{ delta: { content: "B" }, finish_reason: "stop" }], model: "gpt-4o" },
    { choices: [], usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } },
    "data: [DONE]",
  ]);
  const r = await stream({ provider: "openai", apiKey: "k" }, { prompt: "p" });
  assert.equal(r.text, "AB");
  assert.equal(r.usage.totalTokens, 5);
  restore();
});

test("stream: custom adapter one-shot fallback", async () => {
  let n = 0;
  const r = await stream({ provider: "custom", adapter: async (p) => `full:${p}` }, { prompt: "x", onToken: () => n++ });
  assert.equal(n, 1);
  assert.equal(r.text, "full:x");
});
