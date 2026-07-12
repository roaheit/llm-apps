export type LLMProvider = "anthropic" | "openai" | "mistral" | "custom";

export interface LLMConfig {
  provider: LLMProvider;
  apiKey?: string;
  /** Override the default model for the provider. */
  model?: string;
  /** Custom base URL (required when provider is "custom", e.g. a LiteLLM gateway). */
  baseUrl?: string;
  /** Extra headers merged into the request. */
  headers?: Record<string, string>;
}

export type NarrationTone = "narrative" | "analyst" | "casual" | "teacher";

export type SqlDialect =
  | "ansi"
  | "snowflake"
  | "postgres"
  | "mysql"
  | "sqlserver"
  | "bigquery"
  | "oracle";

export interface QueryResults {
  /** Column names, in order. */
  columns: string[];
  /** Row values aligned with `columns`. */
  rows: Array<Array<string | number | boolean | null>>;
  /** Total row count if the sample is truncated. */
  totalRowCount?: number;
}

export interface NarrationRequest {
  sql: string;
  results?: QueryResults;
  dialect?: SqlDialect;
  tone?: NarrationTone;
  /** Extra context, e.g. "this runs nightly against the sales mart". */
  context?: string;
  /** Max rows from `results` to send to the LLM. Default 20. */
  maxSampleRows?: number;
}

export interface Narration {
  /** One-sentence summary of what the query does. */
  summary: string;
  /** Step-by-step plain-English explanation of the query. */
  queryExplanation: string;
  /** What the results mean — only present when results were provided. */
  resultsInterpretation?: string;
  /** Optional caveats: performance smells, ambiguity, dialect quirks. */
  caveats?: string[];
}

export interface NarratorState {
  narration: Narration | null;
  loading: boolean;
  error: string | null;
}
