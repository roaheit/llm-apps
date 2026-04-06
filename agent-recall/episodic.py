"""Episodic (action log) memory layer."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional


class EpisodicMemory:
    """
    Timestamped log of agent actions and outcomes.

    Persisted as newline-delimited JSON (one entry per line) for
    easy streaming and append-only writes.
    """

    def __init__(self, storage_dir: str):
        self.log_path = os.path.join(storage_dir, "episodes.jsonl")
        # Touch the file if it doesn't exist
        if not os.path.exists(self.log_path):
            open(self.log_path, "w").close()

    def log(
        self,
        action: str,
        result: str,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Append an episode to the log."""
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "unix_ts": time.time(),
            "action": action,
            "result": result,
            "tags": tags or [],
            "metadata": metadata or {},
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def _load_all(self) -> list[dict[str, Any]]:
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the n most recent episodes."""
        entries = self._load_all()
        return entries[-n:]

    def filter_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Return all episodes with a specific tag."""
        return [e for e in self._load_all() if tag in e.get("tags", [])]

    def filter_by_action(self, action: str) -> list[dict[str, Any]]:
        """Return all episodes matching a given action name."""
        return [e for e in self._load_all() if e["action"] == action]

    def filter_by_time(self, since_unix: float, until_unix: Optional[float] = None) -> list[dict[str, Any]]:
        """Return episodes within a time range."""
        until_unix = until_unix or time.time()
        return [
            e for e in self._load_all()
            if since_unix <= e.get("unix_ts", 0) <= until_unix
        ]

    def clear(self) -> None:
        """Wipe the episode log."""
        open(self.log_path, "w").close()

    def stats(self) -> dict[str, Any]:
        entries = self._load_all()
        tags: set[str] = set()
        actions: set[str] = set()
        for e in entries:
            tags.update(e.get("tags", []))
            actions.add(e["action"])
        return {
            "total_episodes": len(entries),
            "unique_actions": list(actions),
            "unique_tags": list(tags),
        }
