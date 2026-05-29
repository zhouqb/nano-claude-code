"""Tests for Layer 1 — Budget Reduction (tool_result_budget)."""

from __future__ import annotations

from nano_claude.compaction.tool_result_budget import (
    PER_RESULT_CHAR_CAP,
    ContentReplacementState,
    apply_tool_result_budget,
)


def _msgs(big: str) -> list[dict]:
    return [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": big},
    ]


def test_small_result_untouched():
    state = ContentReplacementState()
    msgs = _msgs("small output")
    out = apply_tool_result_budget(msgs, state, None)
    # Nothing replaced → same list object returned, content intact.
    assert out is msgs
    assert "c1" in state.seen_ids
    assert "c1" not in state.replacements


def test_large_result_spilled_and_previewed(tmp_path):
    state = ContentReplacementState()
    big = "X" * (PER_RESULT_CHAR_CAP + 5_000)
    out = apply_tool_result_budget(_msgs(big), state, tmp_path)

    tool_msg = out[-1]
    assert tool_msg["content"] != big
    assert "truncated by budget" in tool_msg["content"]
    assert "c1" in state.replacements
    # The full content was spilled to a deterministic, tcid-named file.
    spill = tmp_path / "budget-c1.txt"
    assert spill.read_text() == big
    # Non-tool messages are passed through unchanged.
    assert out[0]["content"] == "do a thing"


def test_frozen_decision_is_byte_identical(tmp_path):
    state = ContentReplacementState()
    big = "Y" * (PER_RESULT_CHAR_CAP + 1)

    first = apply_tool_result_budget(_msgs(big), state, tmp_path)[-1]["content"]
    # Second pass re-applies the cached preview (no re-spill, same string).
    second = apply_tool_result_budget(_msgs(big), state, tmp_path)[-1]["content"]
    assert first == second


def test_deterministic_across_fresh_state(tmp_path):
    # Simulates --resume: a brand-new state re-derives the same preview because
    # the spill path is keyed by tool_call_id, not a timestamp/uuid.
    big = "Z" * (PER_RESULT_CHAR_CAP + 1)
    a = apply_tool_result_budget(_msgs(big), ContentReplacementState(), tmp_path)[-1]["content"]
    b = apply_tool_result_budget(_msgs(big), ContentReplacementState(), tmp_path)[-1]["content"]
    assert a == b


def test_seen_but_unreplaced_never_replaced_later(tmp_path):
    # A result first seen while small is frozen as "not replaced"; even if a
    # later pass somehow sees a larger body for the same id, it stays untouched.
    state = ContentReplacementState()
    apply_tool_result_budget(_msgs("tiny"), state, tmp_path)
    assert "c1" in state.seen_ids and "c1" not in state.replacements

    grown = _msgs("W" * (PER_RESULT_CHAR_CAP + 1))
    out = apply_tool_result_budget(grown, state, tmp_path)
    assert out[-1]["content"].startswith("W")  # unchanged, not previewed


def test_no_output_dir_falls_back_to_temp(tmp_path, monkeypatch):
    # With no session output dir, the spill lands in a shared temp dir (still
    # Readable) rather than being dropped.
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    state = ContentReplacementState()
    big = "Q" * (PER_RESULT_CHAR_CAP + 1)
    out = apply_tool_result_budget(_msgs(big), state, None)
    content = out[-1]["content"]
    assert "truncated by budget" in content
    assert (tmp_path / "nano-claude-outputs" / "budget-c1.txt").read_text() == big
