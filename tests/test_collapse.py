"""Tests for Layer 4 — Context Collapse (projection + commit log)."""

from __future__ import annotations

from nano_claude.agent.types import AgentConfig, LoopState
from nano_claude.compaction.collapse import (
    CollapseCommit,
    CollapseState,
    apply_collapses_if_needed,
    project_view,
    reset_collapse,
)


def _read_turn(call_id: str, tool: str = "Read", result: str = "file contents") -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tool}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _read_heavy_conversation(n: int) -> list[dict]:
    msgs: list[dict] = [{"role": "user", "content": "investigate the bug"}]
    for i in range(n):
        msgs += _read_turn(f"r{i}")
    msgs.append({"role": "assistant", "content": "found it"})
    return msgs


async def _summ(_span, _config) -> str:
    return "looked at files r0..rN"


# --- projection -------------------------------------------------------------


def test_project_view_splices_span():
    msgs = _read_heavy_conversation(3)  # r0, r1, r2
    state = CollapseState(commits=[CollapseCommit("cid", "r0", "r2", "summary text")])
    out = project_view(msgs, state)

    # The whole r0..r2 span (6 messages) becomes one placeholder.
    assert len(out) == len(msgs) - 6 + 1
    assert any(
        m.get("role") == "assistant" and "summary text" in (m.get("content") or "") for m in out
    )
    # First/last user/assistant bookends survive.
    assert out[0] == {"role": "user", "content": "investigate the bug"}
    assert out[-1] == {"role": "assistant", "content": "found it"}


def test_project_view_idempotent():
    msgs = _read_heavy_conversation(3)
    state = CollapseState(commits=[CollapseCommit("cid", "r0", "r2", "s")])
    once = project_view(msgs, state)
    twice = project_view(once, state)  # span already gone → no-op
    assert once == twice


def test_project_view_no_commits_is_identity():
    msgs = _read_heavy_conversation(2)
    assert project_view(msgs, CollapseState()) is msgs
    assert project_view(msgs, None) is msgs


# --- apply_collapses_if_needed ----------------------------------------------


async def test_no_collapse_below_threshold():
    config = AgentConfig(context_window=200_000)
    state = LoopState(messages=[], last_input_tokens=100_000)  # below 90% (180k)
    msgs = _read_heavy_conversation(4)
    result = await apply_collapses_if_needed(msgs, state, config, summarize=_summ)
    assert result.committed is False
    assert result.exhausted is False
    assert result.messages is msgs


async def test_collapses_span_over_threshold():
    config = AgentConfig(context_window=200_000)
    state = LoopState(messages=[], last_input_tokens=185_000)  # above 90%
    msgs = _read_heavy_conversation(4)
    result = await apply_collapses_if_needed(msgs, state, config, summarize=_summ)

    assert result.committed is True
    assert len(state.collapse.commits) == 1
    assert any("looked at files" in (m.get("content") or "") for m in result.messages)


async def test_exhausted_when_no_span_left():
    config = AgentConfig(context_window=200_000)
    state = LoopState(messages=[], last_input_tokens=185_000)
    # No read/search span (just a plain exchange) → nothing to collapse.
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = await apply_collapses_if_needed(msgs, state, config, summarize=_summ)
    assert result.committed is False
    assert result.exhausted is True


def test_reset_collapse():
    state = LoopState(
        messages=[], collapse=CollapseState(commits=[CollapseCommit("c", "a", "b", "s")])
    )
    reset_collapse(state)
    assert state.collapse.commits == []
