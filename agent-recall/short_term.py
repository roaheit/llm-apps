"""Short-term (in-session buffer) memory layer."""

from __future__ import annotations

from typing import Any, Optional


class ShortTermMemory:
    """
    In-process buffer for the current session.

    Stores recent messages and observations as plain text entries.
    Auto-reports token count so the Memory class can trigger compression
    when the buffer gets too large.
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self._entries: list[dict[str, Any]] = []

    # Rough token estimate: 1 token ≈ 4 chars
    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def add(self, content: str, metadata: Optional[dict] = None) -> None:
        """Append a new entry to the buffer."""
        self._entries.append({
            "content": content,
            "metadata": metadata or {},
        })

    def get_buffer(self) -> str:
        """Return all buffer entries as a single formatted string."""
        return "\n".join(e["content"] for e in self._entries)

    def is_full(self) -> bool:
        """Return True if the buffer is at or over the token limit."""
        total = sum(self._estimate_tokens(e["content"]) for e in self._entries)
        return total >= self.max_tokens

    def clear(self) -> None:
        """Wipe the buffer."""
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        total_tokens = sum(self._estimate_tokens(e["content"]) for e in self._entries)
        return {
            "entries": len(self._entries),
            "estimated_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "is_full": self.is_full(),
        }
