import { LLMError } from "./types";

/**
 * Extract and parse a JSON value from LLM output that may be wrapped in prose
 * or markdown fences. Strategy: strip fences, try a direct parse, then fall
 * back to scanning for the first balanced `{...}` or `[...]` (string-safe, so
 * braces inside string literals don't miscount).
 *
 * Throws {@link LLMError} if no valid JSON can be recovered.
 */
export function extractJson<T = unknown>(text: string): T {
  const stripped = stripFences(text).trim();

  const direct = tryParse<T>(stripped);
  if (direct.ok) return direct.value;

  const candidate = findFirstJson(stripped);
  if (candidate) {
    const parsed = tryParse<T>(candidate);
    if (parsed.ok) return parsed.value;
  }

  throw new LLMError(`Could not extract JSON from model output: ${text.slice(0, 200)}`);
}

function stripFences(text: string): string {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  return fence ? fence[1] : text;
}

function tryParse<T>(s: string): { ok: true; value: T } | { ok: false } {
  if (!s) return { ok: false };
  try {
    return { ok: true, value: JSON.parse(s) as T };
  } catch {
    return { ok: false };
  }
}

/** Return the substring of the first balanced JSON object/array, or null. */
function findFirstJson(text: string): string | null {
  const start = text.search(/[{[]/);
  if (start < 0) return null;

  const open = text[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === open) depth++;
    else if (ch === close) {
      depth--;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}
