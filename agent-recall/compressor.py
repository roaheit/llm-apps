"""LLM-powered compressor for short-term memory buffer."""

from __future__ import annotations

from typing import Optional


SUMMARIZE_PROMPT = """You are a memory compression assistant. 
Summarize the following agent session buffer into a concise, factual paragraph.
Preserve key facts, decisions, user preferences, and outcomes.
Discard filler, repetition, and low-signal content.

SESSION BUFFER:
{buffer}

SUMMARY:"""


class Compressor:
    """
    Summarises a text buffer into a compact memory using an LLM.

    Default model: gpt-4o-mini (fast, cheap, good enough for compression).
    Can be swapped for any OpenAI-compatible model.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.llm_model = llm_model
        self.api_key = api_key

    def summarize(self, buffer: str) -> str:
        """
        Compress a buffer string into a concise summary.

        Args:
            buffer: Raw text from the short-term memory buffer.

        Returns:
            Compressed summary string.
        """
        if not buffer.strip():
            return ""

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": SUMMARIZE_PROMPT.format(buffer=buffer),
                    }
                ],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            # Fallback: return a truncated version of the buffer
            words = buffer.split()
            if len(words) > 200:
                return " ".join(words[:200]) + " [truncated]"
            return buffer
