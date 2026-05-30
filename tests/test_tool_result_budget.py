"""Tests for Layer 1 — Budget Reduction (per-batch aggregate budget)."""

from __future__ import annotations

from nano_claude.compaction.tool_result_budget import (
    ContentReplacementState,
    apply_tool_result_budget,
)


def _batch(results: list[tuple[str, str]], *, lead: bool = True) -> list[dict]:
    """Build one assistant tool-call turn + its tool-result batch.

    ``results`` is a list of (tool_call_id, content).
    """
    asst = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tcid, "type": "function", "function": {"name": "Bash"}} for tcid, _ in results
        ],
    }
    tools = [{"role": "tool", "tool_call_id": tcid, "content": body} for tcid, body in results]
    head = [{"role": "user", "content": "go"}] if lead else []
    return [*head, asst, *tools]


def _tool_contents(msgs: list[dict]) -> dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in msgs if m.get("role") == "tool"}


# --- under budget -----------------------------------------------------------


def test_batch_under_budget_untouched():
    state = ContentReplacementState()
    msgs = _batch([("c1", "x" * 5_000), ("c2", "y" * 5_000)])  # 10k < 20k
    out = apply_tool_result_budget(msgs, state, None)
    assert out is msgs
    assert state.seen_ids == {"c1", "c2"}
    assert state.replacements == {}


# --- the reported failure mode: aggregate overflow of medium results --------


def test_batch_of_mediums_overflows_and_spills_largest(tmp_path):
    # Six 5k results = 30k > 20k budget; none individually huge. Largest-first
    # eviction spills enough to get back under budget.
    state = ContentReplacementState()
    results = [(f"c{i}", "x" * 5_000) for i in range(6)]
    out = apply_tool_result_budget(_batch(results), state, tmp_path)

    contents = _tool_contents(out)
    spilled = [tcid for tcid, body in contents.items() if "truncated by budget" in body]
    kept = [tcid for tcid, body in contents.items() if "truncated by budget" not in body]
    # Spill 2 of 6 (30k - 2*5k = 20k <= budget); keep 4.
    assert len(spilled) == 2
    assert len(kept) == 4


def test_largest_evicted_first(tmp_path):
    state = ContentReplacementState()
    # Evicting only the biggest brings the batch (26k) under the 20k budget.
    results = [("small", "a" * 2_000), ("mid", "b" * 6_000), ("big", "c" * 18_000)]
    out = apply_tool_result_budget(_batch(results), state, tmp_path)
    contents = _tool_contents(out)
    assert "truncated by budget" in contents["big"]  # evicted
    assert contents["small"] == "a" * 2_000  # kept
    assert contents["mid"] == "b" * 6_000  # kept


def test_single_giant_result_spilled(tmp_path):
    state = ContentReplacementState()
    out = apply_tool_result_budget(_batch([("c1", "z" * 25_000)]), state, tmp_path)
    assert "truncated by budget" in _tool_contents(out)["c1"]
    assert (tmp_path / "budget-c1.txt").read_text() == "z" * 25_000


def test_per_message_split_does_not_dodge_budget(tmp_path):
    # Two separate single-result batches, each 12k (< 20k alone) — must NOT be
    # spilled. Budgeting is per-batch, so independent batches stay independent.
    state = ContentReplacementState()
    msgs = _batch([("a", "x" * 12_000)]) + _batch([("b", "y" * 12_000)], lead=False)
    out = apply_tool_result_budget(msgs, state, tmp_path)
    contents = _tool_contents(out)
    assert "truncated by budget" not in contents["a"]
    assert "truncated by budget" not in contents["b"]


# --- frozen, cache-stable decisions -----------------------------------------


def test_frozen_decision_is_byte_identical(tmp_path):
    state = ContentReplacementState()
    results = [(f"c{i}", "x" * 8_000) for i in range(4)]  # 32k → some spilled
    first = _tool_contents(apply_tool_result_budget(_batch(results), state, tmp_path))
    second = _tool_contents(apply_tool_result_budget(_batch(results), state, tmp_path))
    assert first == second  # re-apply is byte-identical


def test_deterministic_across_fresh_state(tmp_path):
    results = [(f"c{i}", "x" * 8_000) for i in range(4)]
    a = _tool_contents(
        apply_tool_result_budget(_batch(results), ContentReplacementState(), tmp_path)
    )
    b = _tool_contents(
        apply_tool_result_budget(_batch(results), ContentReplacementState(), tmp_path)
    )
    assert a == b  # resume re-derives the same decisions (tcid-keyed spill paths)


def test_seen_but_unreplaced_never_replaced_later(tmp_path):
    # First pass: small batch — everything kept and frozen as seen-unreplaced.
    state = ContentReplacementState()
    apply_tool_result_budget(_batch([("c1", "x" * 3_000)]), state, tmp_path)
    assert "c1" in state.seen_ids and "c1" not in state.replacements
    # Later the same id reappears in an overflowing batch — c1 stays untouched;
    # only the fresh result is eligible to spill.
    msgs = _batch([("c1", "x" * 3_000), ("c2", "y" * 19_000)])  # 22k
    out = apply_tool_result_budget(msgs, state, tmp_path)
    contents = _tool_contents(out)
    assert contents["c1"] == "x" * 3_000
    assert "truncated by budget" in contents["c2"]


# --- preview format ---------------------------------------------------------


def test_default_format_is_prefix(tmp_path):
    state = ContentReplacementState()
    body = "A" * 300 + "B" * 25_000 + "C" * 300
    out = apply_tool_result_budget(_batch([("c1", body)]), state, tmp_path)
    preview = _tool_contents(out)["c1"]
    assert preview.startswith("A" * 250)  # head kept
    assert "C" not in preview  # tail dropped under the prefix default
    assert "middle elided" not in preview


def test_head_tail_format_keeps_both_ends(tmp_path):
    state = ContentReplacementState()
    body = "A" * 300 + "B" * 25_000 + "C" * 300
    out = apply_tool_result_budget(
        _batch([("c1", body)]), state, tmp_path, preview_format="head_tail"
    )
    preview = _tool_contents(out)["c1"]
    assert preview.startswith("A" * 250)  # head kept
    assert "C" * 250 in preview  # tail kept
    assert "middle elided" in preview
    assert "B" not in preview  # middle elided
    # Full content is still on disk regardless of preview shape.
    assert (tmp_path / "budget-c1.txt").read_text() == body


def test_head_tail_decision_is_frozen(tmp_path):
    state = ContentReplacementState()
    body = "A" * 300 + "B" * 25_000 + "C" * 300
    msgs = _batch([("c1", body)])
    first = _tool_contents(
        apply_tool_result_budget(msgs, state, tmp_path, preview_format="head_tail")
    )
    second = _tool_contents(
        apply_tool_result_budget(msgs, state, tmp_path, preview_format="head_tail")
    )
    assert first == second  # re-apply byte-identical, like the prefix path


def test_non_tool_messages_untouched(tmp_path):
    state = ContentReplacementState()
    out = apply_tool_result_budget(_batch([("c1", "z" * 25_000)]), state, tmp_path)
    assert out[0] == {"role": "user", "content": "go"}
    assert out[1]["role"] == "assistant"


def test_no_output_dir_falls_back_to_temp(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    state = ContentReplacementState()
    out = apply_tool_result_budget(_batch([("c1", "q" * 25_000)]), state, None)
    assert "truncated by budget" in _tool_contents(out)["c1"]
    assert (tmp_path / "nano-claude-outputs" / "budget-c1.txt").read_text() == "q" * 25_000
