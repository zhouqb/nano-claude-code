"""Smoke tests for the Phase 1 streaming loop (no tools)."""

from __future__ import annotations

import litellm

from nano_claude.adk.driver import run_turn as query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from tests.conftest import make_acompletion, text_chunk, usage_chunk


async def test_streams_text_and_completes(monkeypatch):
    chunks = [text_chunk("Hello"), text_chunk(", world"), usage_chunk(10, 5)]
    monkeypatch.setattr(litellm, "acompletion", make_acompletion(chunks))

    state = LoopState(messages=[{"role": "user", "content": "hi"}])
    config = AgentConfig()
    seen: list[str] = []

    result = await query_loop(state, config, on_text=seen.append)

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "Hello, world"
    assert seen == ["Hello", ", world"]
    assert result.turn_count == 1


async def test_appends_assistant_message(monkeypatch):
    chunks = [text_chunk("hi there"), usage_chunk(3, 2)]
    monkeypatch.setattr(litellm, "acompletion", make_acompletion(chunks))

    state = LoopState(messages=[{"role": "user", "content": "yo"}])
    await query_loop(state, AgentConfig())

    assert state.messages[-1] == {"role": "assistant", "content": "hi there"}


async def test_tracks_token_usage(monkeypatch):
    chunks = [text_chunk("ok"), usage_chunk(100, 20)]
    monkeypatch.setattr(litellm, "acompletion", make_acompletion(chunks))

    state = LoopState(messages=[{"role": "user", "content": "q"}])
    await query_loop(state, AgentConfig())

    assert state.token_usage.input_tokens == 100
    assert state.token_usage.output_tokens == 20
    assert state.token_usage.total == 120


async def test_max_turns_guard(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", make_acompletion([text_chunk("x")]))

    state = LoopState(messages=[{"role": "user", "content": "q"}], turn_count=5)
    config = AgentConfig(max_turns=5)

    result = await query_loop(state, config)

    assert result.reason is StopReason.MAX_TURNS


async def test_cancel_event_aborts(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", make_acompletion([text_chunk("x")]))

    state = LoopState(messages=[{"role": "user", "content": "q"}])
    state.cancel_event.set()

    result = await query_loop(state, AgentConfig())

    assert result.reason is StopReason.ABORTED


async def test_abort_never_leaves_dangling_tool_calls(tmp_path, monkeypatch):
    """Cancel landing mid-dispatch must not orphan a recorded tool_calls turn.

    Reproduces the timing where the assistant tool_calls message is recorded,
    the user aborts while the tool resolves, and the function-response event is
    dropped — canonical state must still answer every call id, or every later
    request in the session is API-invalid.
    """
    from nano_claude.permissions.manager import PromptOutcome
    from nano_claude.permissions.modes import PermissionMode
    from nano_claude.permissions.settings import Settings
    from tests.conftest import make_sequential_acompletion, tool_call_chunk

    state = LoopState(messages=[{"role": "user", "content": "run it"}])

    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_sequential_acompletion(
            [
                [tool_call_chunk(0, "c1", "Bash", '{"command": "echo hi"}'), usage_chunk(10, 5)],
                [text_chunk("never"), usage_chunk(12, 2)],
            ]
        ),
    )

    async def cancelling_prompter(tool, args, text):
        state.cancel_event.set()  # Esc lands while the permission prompt is up
        return PromptOutcome.ALLOW_ONCE

    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.DEFAULT)
    result = await query_loop(
        state, config, settings=Settings(path=tmp_path / "s.json"), prompter=cancelling_prompter
    )

    assert result.reason is StopReason.ABORTED
    call_ids = {
        tc["id"]
        for m in state.messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    result_ids = {m["tool_call_id"] for m in state.messages if m.get("role") == "tool"}
    assert call_ids, "the assistant tool_calls turn must have been recorded"
    assert call_ids == result_ids  # every call answered — no dangling ids
