export type StoryTone = "narrative" | "analyst" | "casual" | "poetic";

import type { LLMConfig, LLMProvider } from "corellm";
export type { LLMConfig, LLMProvider };

export interface JsonStorytellerProps {
  /**
   * LLM configuration — provider, key, model.
   */
  llm: LLMConfig;

  /**
   * The JSON data to narrate. Accepts a plain object or a JSON string.
   */
  data?: Record<string, unknown> | string;

  /**
   * Storytelling tone for the generated narrative.
   * @default "narrative"
   */
  tone?: StoryTone;

  /**
   * Called when a story is successfully generated.
   */
  onStoryGenerated?: (story: string, tone: StoryTone) => void;

  /**
   * Called when an error occurs during generation.
   */
  onError?: (error: Error) => void;

  /**
   * Custom class name for the root container.
   */
  className?: string;

  /**
   * Inline styles for the root container.
   */
  style?: React.CSSProperties;

  /**
   * If true, renders only the story output — no editor UI.
   * @default false
   */
  headless?: boolean;

  /**
   * Override the default placeholder JSON shown in the editor.
   */
  placeholder?: string;

  /**
   * Dark or light theme.
   * @default "dark"
   */
  theme?: "dark" | "light";
}

export interface UseJsonStorytellerOptions {
  llm: LLMConfig;
  tone?: StoryTone;
  onStoryGenerated?: (story: string, tone: StoryTone) => void;
  onError?: (error: Error) => void;
}

export interface UseJsonStorytellerReturn {
  narrate: (data: Record<string, unknown> | string) => Promise<void>;
  story: string;
  loading: boolean;
  error: Error | null;
  reset: () => void;
}
