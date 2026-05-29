"""Tests for Layer 2 — Snip (structural prune of zombie messages)."""

from __future__ import annotations

from nano_claude.compaction.snip import INTERRUPTED, snip_messages


def _asst_tool(call_ids: list[str]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": c, "type": "function", "function": {}} for c in call_ids],
    }


def test_clean_conversation_untouched():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = snip_messages(msgs)
    assert result.removed == 0
    assert result.messages is msgs  # same object when nothing dropped


def test_fully_interrupted_turn_removed():
    msgs = [
        {"role": "user", "content": "go"},
        _asst_tool(["c1", "c2"]),
        {"role": "tool", "tool_call_id": "c1", "content": INTERRUPTED},
        {"role": "tool", "tool_call_id": "c2", "content": INTERRUPTED},
        {"role": "user", "content": "next"},
    ]
    result = snip_messages(msgs)
    # Assistant turn + both interrupted results gone; user messages remain.
    assert result.removed == 3
    roles = [m["role"] for m in result.messages]
    assert roles == ["user", "user"]
    assert result.tokens_freed > 0
    # No dangling tool_calls left (API-valid).
    assert not any(m.get("tool_calls") for m in result.messages)


def test_partially_interrupted_turn_kept():
    msgs = [
        _asst_tool(["c1", "c2"]),
        {"role": "tool", "tool_call_id": "c1", "content": "real output"},
        {"role": "tool", "tool_call_id": "c2", "content": INTERRUPTED},
    ]
    result = snip_messages(msgs)
    # One real result survives → the whole turn is preserved (conservative).
    assert result.removed == 0


def test_empty_assistant_removed():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "kept"},
    ]
    result = snip_messages(msgs)
    assert result.removed == 2
    assert [m["content"] for m in result.messages if m["role"] == "assistant"] == ["kept"]


def test_interrupted_marker_removed():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": INTERRUPTED},
        {"role": "user", "content": "Continue from where you left off."},
    ]
    result = snip_messages(msgs)
    assert result.removed == 1
    assert all(m.get("content") != INTERRUPTED for m in result.messages)
