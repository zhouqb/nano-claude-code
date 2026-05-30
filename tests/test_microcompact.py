"""Tests for Layer 3 — Microcompact (time-gated clearing of old tool bodies)."""

from __future__ import annotations

from nano_claude.compaction.microcompact import (
    CLEARED_MESSAGE,
    GAP_THRESHOLD_MINUTES,
    microcompact,
)

# A gap comfortably past the threshold → cache presumed cold → clearing allowed.
COLD = GAP_THRESHOLD_MINUTES + 1


def _turn(call_id: str, tool: str, result: str) -> list[dict]:
    """One assistant tool-call + its tool result."""
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _conversation(n: int, tool: str = "Bash") -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "start"}]
    for i in range(n):
        msgs += _turn(f"c{i}", tool, f"output {i}")
    return msgs


# --- time gate --------------------------------------------------------------


def test_no_clear_when_cache_warm():
    # Many old results, but the gap is under threshold → cache still warm → no-op.
    msgs = _conversation(10)
    out = microcompact(msgs, keep_recent=3, gap_minutes=GAP_THRESHOLD_MINUTES - 1)
    assert out is msgs  # same object; nothing cleared


def test_no_clear_when_gap_unknown():
    # gap_minutes=None (no prior assistant timestamp / fresh session) → no-op.
    msgs = _conversation(10)
    out = microcompact(msgs, keep_recent=3, gap_minutes=None)
    assert out is msgs


# --- clearing (cache cold) --------------------------------------------------


def test_keeps_recent_clears_old_when_cold():
    msgs = _conversation(10)  # 10 Bash results
    out = microcompact(msgs, keep_recent=3, gap_minutes=COLD)
    results = [m for m in out if m["role"] == "tool"]
    cleared = [m for m in results if m["content"] == CLEARED_MESSAGE]
    kept = [m for m in results if m["content"] != CLEARED_MESSAGE]
    assert len(cleared) == 7
    assert [m["content"] for m in kept] == ["output 7", "output 8", "output 9"]


def test_noop_when_under_keep_recent():
    msgs = _conversation(3)
    out = microcompact(msgs, keep_recent=6, gap_minutes=COLD)
    assert out is msgs  # same object, nothing to clear even when cold


def test_non_compactable_tools_untouched():
    # A tool not in COMPACTABLE_TOOLS (e.g. a hypothetical "SendMessage") is left
    # alone even when old and cold.
    msgs = [{"role": "user", "content": "x"}]
    for i in range(5):
        msgs += _turn(f"s{i}", "SendMessage", f"sent {i}")
    out = microcompact(msgs, keep_recent=1, gap_minutes=COLD)
    assert all(m["content"].startswith("sent") for m in out if m["role"] == "tool")


def test_idempotent():
    msgs = _conversation(8)
    once = microcompact(msgs, keep_recent=2, gap_minutes=COLD)
    twice = microcompact(once, keep_recent=2, gap_minutes=COLD)
    # Second pass finds nothing new to clear → returns the same object.
    assert twice is once


def test_assistant_and_user_messages_untouched():
    msgs = _conversation(8)
    out = microcompact(msgs, keep_recent=2, gap_minutes=COLD)
    assert out[0] == {"role": "user", "content": "start"}
    assert all(m.get("content") != CLEARED_MESSAGE for m in out if m["role"] == "assistant")
