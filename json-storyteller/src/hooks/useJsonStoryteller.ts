import { useState, useCallback } from "react";
import { callLLM } from "../adapters";
import type { UseJsonStorytellerOptions, UseJsonStorytellerReturn, StoryTone } from "../types";

const TONE_PROMPTS: Record<StoryTone, string> = {
  narrative: "Write an engaging, story-like narrative in 3–5 paragraphs. Give context, flow, and meaning.",
  analyst:   "Write a concise analytical report with clear observations. Paragraph form, no bullet points.",
  casual:    "Explain this like you're texting a friend — short sentences, natural, a bit of personality.",
  poetic:    "Transform this data into lyrical, vivid prose. Find metaphors in the numbers. Make it beautiful.",
};

function buildPrompt(data: string, tone: StoryTone): string {
  return `You are a JSON Storyteller. Read this data and transform it into compelling human-readable text.

Tone: ${tone} — ${TONE_PROMPTS[tone]}

Rules:
- Never mention "JSON", "keys", "objects", or any technical structure terms
- Speak as if describing a real situation, person, or event
- Draw insights and meaning from the values, not the structure
- Keep it under 200 words unless the data is unusually rich
- Pure flowing prose only — no headers, no bullet points

Data:
${data}

Tell the story:`;
}

export function useJsonStoryteller({
  llm,
  tone = "narrative",
  onStoryGenerated,
  onError,
}: UseJsonStorytellerOptions): UseJsonStorytellerReturn {
  const [story, setStory]   = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<Error | null>(null);

  const narrate = useCallback(
    async (data: Record<string, unknown> | string) => {
      setLoading(true);
      setError(null);

      const jsonString = typeof data === "string" ? data : JSON.stringify(data, null, 2);
      const prompt = buildPrompt(jsonString, tone);

      try {
        const text = await callLLM(prompt, llm);
        if (!text) throw new Error("Empty response received. Please try again.");
        setStory(text);
        onStoryGenerated?.(text, tone);
      } catch (err) {
        const e = err instanceof Error ? err : new Error(String(err));
        setError(e);
        onError?.(e);
      } finally {
        setLoading(false);
      }
    },
    [llm, tone, onStoryGenerated, onError]
  );

  const reset = useCallback(() => { setStory(""); setError(null); }, []);

  return { narrate, story, loading, error, reset };
}
