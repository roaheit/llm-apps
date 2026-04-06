"""Tests for agent-recall core functionality (no API calls needed)."""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_recall.layers.short_term import ShortTermMemory
from agent_recall.layers.episodic import EpisodicMemory


# ── Short-term memory ──────────────────────────────────────────────────────────

def test_short_term_add_and_recall():
    mem = ShortTermMemory(max_tokens=1000)
    mem.add("The user prefers Python over JavaScript.")
    mem.add("The project is called agent-recall.")
    buf = mem.get_buffer()
    assert "Python" in buf
    assert "agent-recall" in buf
    print("✓ short_term add + get_buffer")


def test_short_term_clear():
    mem = ShortTermMemory()
    mem.add("Some content")
    mem.clear()
    assert mem.get_buffer() == ""
    print("✓ short_term clear")


def test_short_term_is_full():
    # Very small token limit to trigger quickly
    mem = ShortTermMemory(max_tokens=5)
    mem.add("This is a fairly long sentence that should exceed five tokens easily.")
    assert mem.is_full()
    print("✓ short_term is_full")


def test_short_term_stats():
    mem = ShortTermMemory(max_tokens=500)
    mem.add("hello world")
    stats = mem.stats()
    assert stats["entries"] == 1
    assert "estimated_tokens" in stats
    print("✓ short_term stats")


# ── Episodic memory ────────────────────────────────────────────────────────────

def test_episodic_log_and_recent():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = EpisodicMemory(storage_dir=tmpdir)
        ep.log("web_search", "Found 3 papers", tags=["research"])
        ep.log("write_file", "Wrote summary.md", tags=["output"])
        recent = ep.recent(n=10)
        assert len(recent) == 2
        assert recent[0]["action"] == "web_search"
        print("✓ episodic log + recent")


def test_episodic_filter_by_tag():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = EpisodicMemory(storage_dir=tmpdir)
        ep.log("web_search", "result A", tags=["research"])
        ep.log("code_exec", "result B", tags=["code"])
        ep.log("web_search", "result C", tags=["research"])
        results = ep.filter_by_tag("research")
        assert len(results) == 2
        print("✓ episodic filter_by_tag")


def test_episodic_filter_by_action():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = EpisodicMemory(storage_dir=tmpdir)
        ep.log("web_search", "r1")
        ep.log("write_file", "r2")
        ep.log("web_search", "r3")
        results = ep.filter_by_action("web_search")
        assert len(results) == 2
        print("✓ episodic filter_by_action")


def test_episodic_filter_by_time():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = EpisodicMemory(storage_dir=tmpdir)
        before = time.time()
        ep.log("action_a", "early")
        time.sleep(0.05)
        mid = time.time()
        ep.log("action_b", "late")
        results = ep.filter_by_time(since_unix=mid)
        assert len(results) == 1
        assert results[0]["action"] == "action_b"
        print("✓ episodic filter_by_time")


def test_episodic_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = EpisodicMemory(storage_dir=tmpdir)
        ep.log("a", "b")
        ep.clear()
        assert ep.recent() == []
        print("✓ episodic clear")


def test_episodic_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        ep = EpisodicMemory(storage_dir=tmpdir)
        ep.log("web_search", "r1", tags=["research"])
        ep.log("code_exec", "r2", tags=["code"])
        stats = ep.stats()
        assert stats["total_episodes"] == 2
        assert "web_search" in stats["unique_actions"]
        print("✓ episodic stats")


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nRunning agent-recall tests (no API key needed)\n")
    test_short_term_add_and_recall()
    test_short_term_clear()
    test_short_term_is_full()
    test_short_term_stats()
    test_episodic_log_and_recent()
    test_episodic_filter_by_tag()
    test_episodic_filter_by_action()
    test_episodic_filter_by_time()
    test_episodic_clear()
    test_episodic_stats()
    print("\nAll tests passed ✓\n")
