"""Integration: auto-compaction and the circuit breaker inside query_loop."""

from __future__ import annotations

import json

import litellm

from nano_claude.agent import loop as loop_module
from nano_claude.agent.loop import LoopCallbacks, query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.permissions.manager import PromptOutcome
from nano_claude.permissions.settings import Settings
from tests.conftest import (
    make_sequential_acompletion,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)


async def _allow(tool, args, text):
    return PromptOutcome.ALLOW_ONCE


def _read_call(tmp_path, call_id: str):
    target = tmp_path / "f.txt"
    if not target.exists():
        target.write_text("file contents")
    args = json.dumps({"file_path": str(target)})
    return tool_call_chunk(0, call_id, "Read", args)


async def test_auto_compact_fires_mid_loop(tmp_path, monkeypatch):
    # turn 1 reports a near-full context (high prompt_tokens) and calls a tool,
    # so the loop continues; turn 2's pre-call gate should auto-compact.
    streams = [
        [_read_call(tmp_path, "c1"), usage_chunk(99_000, 50)],  # turn 1
        [text_chunk("COMPACTED SUMMARY"), usage_chunk(10, 10)],  # summary call
        [text_chunk("all done"), usage_chunk(20, 5)],  # turn 2
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    compacted = {"fired": False}
    callbacks = LoopCallbacks(on_compact=lambda: compacted.__setitem__("fired", True))

    state = LoopState(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
    )
    config = AgentConfig(cwd=str(tmp_path), context_window=100_000)  # threshold 87k

    result = await query_loop(
        state,
        config,
        settings=Settings(path=tmp_path / "s.json"),
        prompter=_allow,
        callbacks=callbacks,
    )

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "all done"
    assert compacted["fired"]
    # After compaction: [system, summary, ...recent tail...]
    assert state.messages[0]["role"] == "system"
    assert "COMPACTED SUMMARY" in state.messages[1]["content"]
    assert state.last_input_tokens == 20  # reset by compaction, then set by turn 2


async def test_circuit_breaker_disables_auto_compact(tmp_path, monkeypatch):
    # Make every compaction attempt fail; after MAX failures auto-compact is off.
    async def _failing_compact(state, config):
        state.consecutive_compact_failures += 1
        return False

    monkeypatch.setattr(loop_module, "compact_conversation", _failing_compact)

    streams = [
        [_read_call(tmp_path, "c1"), usage_chunk(99_000, 5)],
        [_read_call(tmp_path, "c2"), usage_chunk(99_000, 5)],
        [_read_call(tmp_path, "c3"), usage_chunk(99_000, 5)],
        [text_chunk("done"), usage_chunk(99_000, 5)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    disabled = {"fired": False}
    callbacks = LoopCallbacks(on_compact_disabled=lambda: disabled.__setitem__("fired", True))

    state = LoopState(
        messages=[{"role": "user", "content": "go"}],
        last_input_tokens=99_000,  # over threshold from the start
    )
    config = AgentConfig(cwd=str(tmp_path), context_window=100_000)

    result = await query_loop(
        state,
        config,
        settings=Settings(path=tmp_path / "s.json"),
        prompter=_allow,
        callbacks=callbacks,
    )

    assert result.reason is StopReason.COMPLETED
    assert config.auto_compact is False
    assert disabled["fired"]
    assert state.consecutive_compact_failures >= 3
