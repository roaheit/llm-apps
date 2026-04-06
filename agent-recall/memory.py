"""
agent-recall: Drop-in persistent memory for LLM agents.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .layers.short_term import ShortTermMemory
from .layers.long_term import LongTermMemory
from .layers.episodic import EpisodicMemory
from .compressor import Compressor


class Memory:
    """
    Three-layer memory system for LLM agents.

    Layers:
        short_term  — In-session buffer, auto-compresses when token limit hit.
        long_term   — Cross-session semantic store, retrieved via similarity search.
        episodic    — Timestamped action/outcome log, queryable by time or tag.

    Example::

        mem = Memory(agent_id="research-agent")
        mem.remember("User prefers bullet points", layer="long_term")
        context = mem.recall("What are the user's preferences?")
        mem.log_episode(action="web_search", result="Found 3 papers", tags=["research"])
    """

    def __init__(
        self,
        agent_id: str,
        storage_dir: str = ".agent_recall",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        max_short_term_tokens: int = 2000,
        top_k_recall: int = 5,
        api_key: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.storage_dir = os.path.join(storage_dir, agent_id)
        os.makedirs(self.storage_dir, exist_ok=True)

        api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.short_term = ShortTermMemory(
            max_tokens=max_short_term_tokens,
        )
        self.long_term = LongTermMemory(
            agent_id=agent_id,
            storage_dir=self.storage_dir,
            embedding_model=embedding_model,
            api_key=api_key,
            top_k=top_k_recall,
        )
        self.episodic = EpisodicMemory(
            storage_dir=self.storage_dir,
        )
        self.compressor = Compressor(
            llm_model=llm_model,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def remember(self, content: str, layer: str = "long_term", metadata: Optional[dict] = None) -> None:
        """
        Store a memory in the specified layer.

        Args:
            content:  The text to remember.
            layer:    One of "short_term", "long_term", or "episodic".
            metadata: Optional dict of extra fields (tags, source, etc.).
        """
        if layer == "short_term":
            self.short_term.add(content, metadata=metadata)
        elif layer == "long_term":
            self.long_term.add(content, metadata=metadata)
        elif layer == "episodic":
            raise ValueError("Use log_episode() to write to episodic memory.")
        else:
            raise ValueError(f"Unknown layer '{layer}'. Choose short_term, long_term, or episodic.")

    def recall(self, query: str, layers: Optional[list[str]] = None) -> str:
        """
        Retrieve relevant memories for a query, across specified layers.

        Args:
            query:  Natural language query.
            layers: Layers to search. Defaults to ["short_term", "long_term"].

        Returns:
            A formatted string of relevant memories, ready to inject into a prompt.
        """
        if layers is None:
            layers = ["short_term", "long_term"]

        sections = []

        if "short_term" in layers:
            buf = self.short_term.get_buffer()
            if buf:
                sections.append(f"[Recent context]\n{buf}")

        if "long_term" in layers:
            results = self.long_term.search(query)
            if results:
                joined = "\n".join(f"- {r}" for r in results)
                sections.append(f"[Relevant long-term memories]\n{joined}")

        if "episodic" in layers:
            episodes = self.episodic.recent(n=5)
            if episodes:
                joined = "\n".join(
                    f"- [{e['timestamp']}] {e['action']}: {e['result']}" for e in episodes
                )
                sections.append(f"[Recent actions]\n{joined}")

        return "\n\n".join(sections) if sections else ""

    def log_episode(
        self,
        action: str,
        result: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log an agent action and its outcome to episodic memory.

        Args:
            action:   What the agent did (e.g. "web_search", "write_file").
            result:   What happened (e.g. "Found 3 relevant papers").
            tags:     Optional list of tags for filtering later.
            metadata: Any extra structured data to store alongside.
        """
        self.episodic.log(action=action, result=result, tags=tags or [], metadata=metadata or {})

    def compress(self) -> str:
        """
        Summarize and compress the short-term buffer using an LLM.
        Moves the summary to long-term memory and clears the buffer.

        Returns:
            The generated summary string.
        """
        buffer = self.short_term.get_buffer()
        if not buffer:
            return ""

        summary = self.compressor.summarize(buffer)
        self.long_term.add(f"[Session summary] {summary}", metadata={"type": "compressed_session"})
        self.short_term.clear()
        return summary

    def clear(self, layer: Optional[str] = None) -> None:
        """
        Clear one or all memory layers.

        Args:
            layer: Layer to clear. If None, clears all layers.
        """
        if layer is None or layer == "short_term":
            self.short_term.clear()
        if layer is None or layer == "long_term":
            self.long_term.clear()
        if layer is None or layer == "episodic":
            self.episodic.clear()

    def stats(self) -> dict[str, Any]:
        """Return a summary of current memory state across all layers."""
        return {
            "agent_id": self.agent_id,
            "short_term": self.short_term.stats(),
            "long_term": self.long_term.stats(),
            "episodic": self.episodic.stats(),
        }
