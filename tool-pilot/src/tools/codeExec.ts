import type { ToolDefinition } from "../types";

export const codeExec: ToolDefinition = {
  name: "code_exec",
  description:
    "Execute a JavaScript code snippet in a sandboxed environment and return the result. " +
    "The code runs via Function constructor. Use console.log() for output — all logged values are captured and returned. " +
    "The last expression value is also returned.",
  parameters: [
    {
      name: "code",
      type: "string",
      description: "JavaScript code to execute",
      required: true,
    },
  ],
  execute: async (args) => {
    const code = args.code as string;
    if (!code) return "Error: code is required";

    const logs: string[] = [];
    const sandbox = {
      console: {
        log: (...a: unknown[]) => logs.push(a.map(String).join(" ")),
        warn: (...a: unknown[]) => logs.push(`[warn] ${a.map(String).join(" ")}`),
        error: (...a: unknown[]) => logs.push(`[error] ${a.map(String).join(" ")}`),
      },
      Math,
      Date,
      JSON,
      parseInt,
      parseFloat,
      isNaN,
      isFinite,
      encodeURIComponent,
      decodeURIComponent,
      Array,
      Object,
      String: globalThis.String,
      Number: globalThis.Number,
      Boolean: globalThis.Boolean,
      Map,
      Set,
      RegExp,
      Promise,
    };

    try {
      const keys = Object.keys(sandbox);
      const vals = Object.values(sandbox);
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      const fn = new Function(...keys, `"use strict";\n${code}`);
      const result = await fn(...vals);

      const parts: string[] = [];
      if (logs.length > 0) parts.push(logs.join("\n"));
      if (result !== undefined) parts.push(`→ ${JSON.stringify(result)}`);
      return parts.length > 0 ? parts.join("\n") : "(no output)";
    } catch (e) {
      return `Execution error: ${(e as Error).message}`;
    }
  },
};
