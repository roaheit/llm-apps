import type { ToolDefinition } from "../types";

export interface CodeExecOptions {
  /**
   * Required acknowledgement. `code_exec` runs model-generated JavaScript in the
   * HOST realm via the `Function` constructor — it is **not** a sandbox. Executed
   * code can reach ambient globals (`globalThis`, `fetch`, …) through the
   * prototype chain and act with the page's privileges. Only enable it for fully
   * trusted input. Tool creation throws unless this is `true`.
   */
  acknowledgeUnsafe: boolean;
  /**
   * Best-effort wall-clock limit (ms). NOTE: this can only interrupt code that
   * yields (awaits) — a synchronous infinite loop still blocks the thread. For a
   * hard time/CPU boundary, run untrusted code in a Web Worker or a wasm
   * interpreter instead. @default 5000
   */
  timeoutMs?: number;
}

/**
 * Create a `code_exec` tool that evaluates a JavaScript snippet.
 *
 * ⚠️ SECURITY: this executes arbitrary, model-generated code in the host
 * environment with **no real sandbox**. The curated globals passed into the
 * function are a convenience, not a security boundary — they do not prevent
 * access to ambient globals. Never enable this for untrusted prompts or input.
 * You must pass `acknowledgeUnsafe: true`.
 */
export function createCodeExecTool(options: CodeExecOptions): ToolDefinition {
  if (!options?.acknowledgeUnsafe) {
    throw new Error(
      "createCodeExecTool: refusing to create a code-execution tool without " +
        "`acknowledgeUnsafe: true`. code_exec runs model-generated JavaScript in the host " +
        "realm with no sandbox — only enable it for fully trusted input."
    );
  }
  const timeoutMs = options.timeoutMs ?? 5000;

  return {
    name: "code_exec",
    description:
      "Execute a JavaScript code snippet and return the result. " +
      "WARNING: runs in the host environment with NO sandbox — only safe for trusted input. " +
      "Use console.log() for output; the last expression value is also returned.",
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
      // Curated globals — a convenience for the executed snippet, NOT a security
      // boundary. Ambient globals remain reachable; do not rely on this list.
      const scope = {
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
        const keys = Object.keys(scope);
        const vals = Object.values(scope);
        const fn = new Function(...keys, `"use strict";\n${code}`);

        let timer: ReturnType<typeof setTimeout> | undefined;
        const timeout = new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error(`code_exec timed out after ${timeoutMs}ms`)), timeoutMs);
        });

        let result: unknown;
        try {
          result = await Promise.race([Promise.resolve().then(() => fn(...vals)), timeout]);
        } finally {
          if (timer) clearTimeout(timer);
        }

        const parts: string[] = [];
        if (logs.length > 0) parts.push(logs.join("\n"));
        if (result !== undefined) parts.push(`→ ${JSON.stringify(result)}`);
        return parts.length > 0 ? parts.join("\n") : "(no output)";
      } catch (e) {
        return `Execution error: ${(e as Error).message}`;
      }
    },
  };
}
