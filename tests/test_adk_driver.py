"""Driver-level behavior tests: wire fidelity, gates, accounting, serialization.

These drive ``run_turn`` through the full production stack (Runner → LlmAgent
→ LiteLlm) with ``litellm.acompletion`` faked, and assert on what actually
reaches the wire — the request ``messages`` kwarg — plus the turn gates and
usage accounting that moved into the ADK callbacks.
"""

from __future__ import annotations

import asyncio
import json

import litellm
import pytest

from nano_claude.adk.driver import run_turn
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.permissions.manager import PromptOutcome
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from tests.conftest import (
    FakeStream,
    make_sequential_acompletion,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)


async def _allow(tool, args, text):
    return PromptOutcome.ALLOW_ONCE


def _config(tmp_path, **kwargs) -> AgentConfig:
    kwargs.setdefault("cwd", str(tmp_path))
    kwargs.setdefault("permission_mode", PermissionMode.BYPASS)
    return AgentConfig(**kwargs)


def _settings(tmp_path) -> Settings:
    return Settings(path=tmp_path / "s.json")


async def test_request_wire_format_matches_old_loop(tmp_path, monkeypatch):
    """The OpenAI messages litellm receives are exactly the old loop's format."""
    captured: list[list[dict]] = []

    streams = iter(
        [
            FakeStream(
                [tool_call_chunk(0, "call_1", "Bash", '{"command": "true"}'), usage_chunk(10, 5)]
            ),
            FakeStream([text_chunk("done"), usage_chunk(12, 2)]),
        ]
    )

    async def capturing_acompletion(*args, **kwargs):
        captured.append(kwargs["messages"])
        return next(streams)

    monkeypatch.setattr(litellm, "acompletion", capturing_acompletion)

    state = LoopState(
        messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "run true"},
        ]
    )
    result = await run_turn(state, _config(tmp_path), settings=_settings(tmp_path))

    assert result.reason is StopReason.COMPLETED
    # First request: system + user, byte-identical.
    assert captured[0][0] == {"role": "system", "content": "be brief"}
    assert captured[0][1] == {"role": "user", "content": "run true"}
    # Second request adds the assistant tool_call turn and the raw tool result.
    roles = [m["role"] for m in captured[1]]
    assert roles == ["system", "user", "assistant", "tool"]
    assistant = captured[1][2]
    (tc,) = assistant["tool_calls"]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "Bash"
    assert json.loads(tc["function"]["arguments"]) == {"command": "true"}
    tool_msg = captured[1][3]
    assert tool_msg["tool_call_id"] == "call_1"
    assert isinstance(tool_msg["content"], str)
    assert not tool_msg["content"].startswith("{")  # raw string, not JSON-wrapped


async def test_max_turns_gate_fires_before_model_call(tmp_path, monkeypatch):
    calls = []

    async def counting_acompletion(*args, **kwargs):
        calls.append(1)
        return FakeStream([text_chunk("x")])

    monkeypatch.setattr(litellm, "acompletion", counting_acompletion)

    state = LoopState(messages=[{"role": "user", "content": "hi"}], turn_count=3)
    result = await run_turn(state, _config(tmp_path, max_turns=3), settings=_settings(tmp_path))

    assert result.reason is StopReason.MAX_TURNS
    assert calls == []  # gate fired pre-request, like the old loop


async def test_usage_and_turn_count_accumulate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_sequential_acompletion(
            [
                [tool_call_chunk(0, "c1", "Bash", '{"command": "true"}'), usage_chunk(100, 10)],
                [text_chunk("ok"), usage_chunk(150, 5)],
            ]
        ),
    )

    state = LoopState(messages=[{"role": "user", "content": "go"}])
    result = await run_turn(state, _config(tmp_path), settings=_settings(tmp_path))

    assert result.reason is StopReason.COMPLETED
    assert state.turn_count == 2  # one per LLM call
    assert state.token_usage.input_tokens == 250
    assert state.token_usage.output_tokens == 15
    # Compaction's context signal tracks the most recent request.
    assert state.last_input_tokens == 150


async def test_cancel_mid_stream_aborts_without_further_model_calls(tmp_path, monkeypatch):
    """Cancel during a streamed response → ABORTED, and no follow-up request.

    Note: ADK's Runner drives the agent in a producer task, so a fully
    buffered fake stream may complete the in-flight response before the
    consumer observes the cancel — the contract asserted here is prompt
    ABORTED reporting and that no *new* LLM request is issued after cancel.
    """
    state = LoopState(messages=[{"role": "user", "content": "hi"}])
    requests_after_cancel = []

    streams = iter(
        [
            FakeStream(
                [tool_call_chunk(0, "c1", "Bash", '{"command": "true"}'), usage_chunk(10, 5)]
            ),
            FakeStream([text_chunk("never"), usage_chunk(12, 2)]),
        ]
    )

    async def _acompletion(*args, **kwargs):
        if state.cancel_event.is_set():
            requests_after_cancel.append(1)
        else:
            state.cancel_event.set()  # Ctrl-C lands during the first request
        return next(streams)

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    result = await run_turn(state, _config(tmp_path), settings=_settings(tmp_path))

    assert result.reason is StopReason.ABORTED
    # The before_model gate stops the loop: no request was issued post-cancel.
    assert requests_after_cancel == []
    # The pending tool call was answered with the interrupt sentinel (or not
    # executed at all) — never with real output.
    for m in state.messages:
        if m.get("role") == "tool":
            assert m["content"] == "[Interrupted]"


async def test_concurrent_permission_prompts_are_serialized(tmp_path, monkeypatch):
    """Two parallel 'ask' tool calls must prompt one at a time (prompt lock)."""
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_sequential_acompletion(
            [
                [
                    tool_call_chunk(0, "c1", "Bash", '{"command": "echo one"}'),
                    tool_call_chunk(1, "c2", "Bash", '{"command": "echo two"}'),
                    usage_chunk(10, 5),
                ],
                [text_chunk("both ran"), usage_chunk(12, 2)],
            ]
        ),
    )

    in_prompt = False
    overlapped = False
    prompts = []

    async def prompter(tool, args, text):
        nonlocal in_prompt, overlapped
        if in_prompt:
            overlapped = True
        in_prompt = True
        await asyncio.sleep(0.01)  # widen the race window
        in_prompt = False
        prompts.append(text)
        return PromptOutcome.ALLOW_ONCE

    state = LoopState(messages=[{"role": "user", "content": "run both"}])
    config = _config(tmp_path, permission_mode=PermissionMode.DEFAULT)
    result = await run_turn(state, config, settings=_settings(tmp_path), prompter=prompter)

    assert result.reason is StopReason.COMPLETED
    assert len(prompts) == 2
    assert not overlapped
    tool_results = [m["content"] for m in state.messages if m.get("role") == "tool"]
    assert sorted(tool_results) == ["one", "two"]


@pytest.mark.parametrize("auto_compact", [False])
async def test_blocked_gate_skips_model_and_reports_notice(tmp_path, monkeypatch, auto_compact):
    calls = []

    async def counting_acompletion(*args, **kwargs):
        calls.append(1)
        return FakeStream([text_chunk("x")])

    monkeypatch.setattr(litellm, "acompletion", counting_acompletion)

    state = LoopState(messages=[{"role": "user", "content": "hi"}])
    state.last_input_tokens = 999_999  # context "full"
    config = _config(tmp_path, auto_compact=auto_compact, context_window=100_000)
    result = await run_turn(state, config, settings=_settings(tmp_path))

    assert result.reason is StopReason.BLOCKED
    assert "compact" in result.final_text.lower()
    assert calls == []


async def test_view_differs_from_canonical_after_spill(tmp_path, monkeypatch):
    """Layer 1 spill shapes the wire view; state.messages stays canonical."""
    captured: list[list[dict]] = []
    big_output = "x" * 200_000  # far past the per-result budget

    streams = iter(
        [
            FakeStream(
                [tool_call_chunk(0, "c1", "Bash", '{"command": "true"}'), usage_chunk(10, 5)]
            ),
            FakeStream([text_chunk("done"), usage_chunk(12, 2)]),
            FakeStream([text_chunk("again"), usage_chunk(14, 2)]),
        ]
    )

    async def capturing_acompletion(*args, **kwargs):
        captured.append(kwargs["messages"])
        return next(streams)

    monkeypatch.setattr(litellm, "acompletion", capturing_acompletion)

    from nano_claude.session.storage import SessionStorage, session_file

    storage = SessionStorage(session_file(str(tmp_path), "sid", root=tmp_path / "root"), "sid")
    state = LoopState(messages=[{"role": "user", "content": "go"}], storage=storage)
    config = _config(tmp_path)
    result = await run_turn(state, config, settings=_settings(tmp_path))
    assert result.reason is StopReason.COMPLETED

    # Force a huge canonical tool result, then run another turn. Budget
    # decisions are frozen per tool_call_id, so reset them as a session
    # resume would (LoopState starts with budget=None).
    for msg in state.messages:
        if msg.get("role") == "tool":
            msg["content"] = big_output
    state.budget = None
    state.messages.append({"role": "user", "content": "next"})
    result = await run_turn(state, config, settings=_settings(tmp_path))
    assert result.reason is StopReason.COMPLETED

    wire_tool = next(m for m in captured[2] if m["role"] == "tool")
    canonical_tool = next(m for m in state.messages if m.get("role") == "tool")
    assert canonical_tool["content"] == big_output  # canonical untouched
    assert len(wire_tool["content"]) < len(big_output)  # view was budgeted
