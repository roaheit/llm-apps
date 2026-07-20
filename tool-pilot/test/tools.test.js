const test = require("node:test");
const assert = require("node:assert/strict");
const { createCodeExecTool, fileRead } = require("../dist/index.js");

test("createCodeExecTool: refuses without acknowledgeUnsafe", () => {
  assert.throws(() => createCodeExecTool({ acknowledgeUnsafe: false }), /acknowledgeUnsafe/);
  assert.throws(() => createCodeExecTool({}), /acknowledgeUnsafe/);
});

test("createCodeExecTool: honest description + executes when acknowledged", async () => {
  const tool = createCodeExecTool({ acknowledgeUnsafe: true });
  assert.equal(tool.name, "code_exec");
  assert.match(tool.description, /NO sandbox/);
  assert.doesNotMatch(tool.description, /sandboxed environment/i);
  const out = await tool.execute({ code: "console.log('hi'); return 1 + 1;" });
  assert.match(out, /hi/);
  assert.match(out, /→ 2/);
});

test("createCodeExecTool: best-effort timeout on async code", async () => {
  const tool = createCodeExecTool({ acknowledgeUnsafe: true, timeoutMs: 100 });
  const out = await tool.execute({ code: "return new Promise(function () {});" });
  assert.match(out, /timed out after 100ms/);
});

test("fileRead: SSRF guard blocks loopback/private/link-local hosts", async () => {
  const blocked = [
    "http://localhost:9999/",
    "http://127.0.0.1/",
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    "http://169.254.169.254/latest/meta-data",
  ];
  for (const p of blocked) {
    assert.match(await fileRead.execute({ path: p }), /^Blocked/, p);
  }
});

test("fileRead: non-http path is not supported", async () => {
  assert.match(await fileRead.execute({ path: "/etc/passwd" }), /not supported/);
});

test("fileRead: public host passes the guard, fetches + strips HTML", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response("<p>Hello <b>world</b></p>", { status: 200 });
  assert.equal(await fileRead.execute({ path: "https://example.com/page" }), "Hello world");
  globalThis.fetch = realFetch;
});
