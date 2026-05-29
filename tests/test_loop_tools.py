"""Integration tests for the Phase 2 loop: tool dispatch end-to-end."""

from __future__ import annotations

import json

import litellm

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.permissions.manager import PromptOutcome
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from tests.conftest import (
    make_sequential_acompletion,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)


async def _allow_once(tool, args, text):
    return PromptOutcome.ALLOW_ONCE


async def _deny_once(tool, args, text):
    return PromptOutcome.DENY_ONCE


def _config(tmp_path) -> AgentConfig:
    return AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.DEFAULT)


async def test_edit_file_end_to_end(tmp_path, monkeypatch):
    target = tmp_path / "greeting.txt"
    target.write_text("hello world")

    edit_args = json.dumps({"file_path": str(target), "old_string": "world", "new_string": "there"})
    streams = [
        [tool_call_chunk(0, "call_1", "Edit", edit_args), usage_chunk(10, 5)],
        [text_chunk("Done."), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "fix it"}])
    settings = Settings(path=tmp_path / "s.json")

    result = await query_loop(state, _config(tmp_path), settings=settings, prompter=_allow_once)

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "Done."
    assert target.read_text() == "hello there"

    # Invariant: the assistant tool_call is followed by a matching tool message.
    assistant = next(m for m in state.messages if m.get("tool_calls"))
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "Replaced 1 occurrence" in tool_msg["content"]


async def test_denied_tool_does_not_run(tmp_path, monkeypatch):
    target = tmp_path / "greeting.txt"
    target.write_text("hello world")

    edit_args = json.dumps({"file_path": str(target), "old_string": "world", "new_string": "there"})
    streams = [
        [tool_call_chunk(0, "call_1", "Edit", edit_args), usage_chunk(10, 5)],
        [text_chunk("Okay, leaving it."), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "fix it"}])
    settings = Settings(path=tmp_path / "s.json")

    result = await query_loop(state, _config(tmp_path), settings=settings, prompter=_deny_once)

    assert result.reason is StopReason.COMPLETED
    assert target.read_text() == "hello world"  # unchanged
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "Permission denied" in tool_msg["content"]


async def test_read_tool_auto_allowed(tmp_path, monkeypatch):
    target = tmp_path / "data.txt"
    target.write_text("line one\nline two")

    read_args = json.dumps({"file_path": str(target)})
    streams = [
        [tool_call_chunk(0, "call_1", "Read", read_args), usage_chunk(10, 5)],
        [text_chunk("It has two lines."), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "read it"}])
    settings = Settings.load(tmp_path / "s.json")  # defaults allow Read

    async def _never(tool, args, text):
        raise AssertionError("Read should not prompt")

    result = await query_loop(state, _config(tmp_path), settings=settings, prompter=_never)

    assert result.reason is StopReason.COMPLETED
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "line one" in tool_msg["content"]


async def test_unknown_tool_returns_error_message(tmp_path, monkeypatch):
    streams = [
        [tool_call_chunk(0, "call_1", "Bogus", "{}"), usage_chunk(10, 5)],
        [text_chunk("Sorry."), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "do x"}])
    settings = Settings(path=tmp_path / "s.json")

    async def _never(tool, args, text):
        raise AssertionError("unknown tool should not prompt")

    result = await query_loop(state, _config(tmp_path), settings=settings, prompter=_never)

    assert result.reason is StopReason.COMPLETED
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "unknown tool" in tool_msg["content"]


async def test_invalid_json_args_returns_error(tmp_path, monkeypatch):
    streams = [
        [tool_call_chunk(0, "call_1", "Read", "{not json"), usage_chunk(10, 5)],
        [text_chunk("recovered"), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "go"}])
    settings = Settings(path=tmp_path / "s.json")

    result = await query_loop(state, _config(tmp_path), settings=settings, prompter=_deny_once)

    assert result.reason is StopReason.COMPLETED
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "not valid JSON" in tool_msg["content"]
