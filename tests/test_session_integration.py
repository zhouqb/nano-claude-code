"""Integration: the loop persists to storage, and sessions resume cleanly."""

from __future__ import annotations

import json

import litellm

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.permissions.manager import PromptOutcome
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from nano_claude.session.restore import load_session, restore_read_file_state
from nano_claude.session.storage import SessionStorage, session_file
from tests.conftest import (
    make_sequential_acompletion,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)


async def _allow(tool, args, text):
    return PromptOutcome.ALLOW_ONCE


def _new_storage(tmp_path) -> SessionStorage:
    path = session_file(str(tmp_path), "sid", root=tmp_path / "root")
    return SessionStorage(path, "sid")


async def test_loop_persists_full_turn(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("hello world")
    read_args = json.dumps({"file_path": str(target)})
    edit_args = json.dumps({"file_path": str(target), "old_string": "world", "new_string": "there"})
    streams = [
        [tool_call_chunk(0, "call_0", "Read", read_args), usage_chunk(10, 5)],
        [tool_call_chunk(0, "call_1", "Edit", edit_args), usage_chunk(10, 5)],
        [text_chunk("Done."), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    storage = _new_storage(tmp_path)
    state = LoopState(storage=storage)
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.DEFAULT)

    # main appends + persists the user message; mimic that.
    user_msg = {"role": "user", "content": "fix it"}
    state.messages.append(user_msg)
    storage.append_message(user_msg)

    result = await query_loop(
        state, config, settings=Settings(path=tmp_path / "s.json"), prompter=_allow
    )
    await storage.flush()

    assert result.reason is StopReason.COMPLETED

    # A clean session round-trips to exactly the in-memory messages.
    restored = load_session(storage.path)
    assert restored == state.messages
    roles = [m["role"] for m in restored]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


async def test_resume_appends_to_same_file(tmp_path, monkeypatch):
    streams_first = [[text_chunk("first reply"), usage_chunk(2, 1)]]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams_first))

    storage = _new_storage(tmp_path)
    state = LoopState(storage=storage)
    config = AgentConfig(cwd=str(tmp_path))

    msg = {"role": "user", "content": "hello"}
    state.messages.append(msg)
    storage.append_message(msg)
    await query_loop(state, config, settings=Settings(path=tmp_path / "s.json"), prompter=_allow)
    await storage.flush()

    path = storage.path

    # --- "restart": resume from disk into a fresh state ---
    streams_second = [[text_chunk("second reply"), usage_chunk(2, 1)]]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams_second))

    resumed_storage = SessionStorage(path, "sid")
    resumed = LoopState(messages=load_session(path), storage=resumed_storage)
    assert [m["content"] for m in resumed.messages] == ["hello", "first reply"]

    msg2 = {"role": "user", "content": "again"}
    resumed.messages.append(msg2)
    resumed_storage.append_message(msg2)
    await query_loop(resumed, config, settings=Settings(path=tmp_path / "s.json"), prompter=_allow)
    await resumed_storage.flush()

    final = load_session(path)
    assert [m["content"] for m in final] == [
        "hello",
        "first reply",
        "again",
        "second reply",
    ]


async def test_resume_restores_read_state_for_existing_file_edit(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("hello world")

    read_args = json.dumps({"file_path": str(target)})
    streams_first = [
        [tool_call_chunk(0, "call_read", "Read", read_args), usage_chunk(10, 5)],
        [text_chunk("read done"), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams_first))

    storage = _new_storage(tmp_path)
    state = LoopState(storage=storage)
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.DEFAULT)
    msg = {"role": "user", "content": "read f"}
    state.messages.append(msg)
    storage.append_message(msg)
    await query_loop(state, config, settings=Settings(path=tmp_path / "s.json"), prompter=_allow)
    await storage.flush()

    edit_args = json.dumps({"file_path": str(target), "old_string": "world", "new_string": "there"})
    streams_second = [
        [tool_call_chunk(0, "call_edit", "Edit", edit_args), usage_chunk(10, 5)],
        [text_chunk("edited"), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams_second))

    resumed_storage = SessionStorage(storage.path, "sid")
    resumed_messages = load_session(storage.path)
    resumed = LoopState(
        messages=resumed_messages,
        storage=resumed_storage,
        read_file_state=restore_read_file_state(resumed_messages, str(tmp_path)),
    )
    msg2 = {"role": "user", "content": "edit it"}
    resumed.messages.append(msg2)
    resumed_storage.append_message(msg2)
    result = await query_loop(
        resumed, config, settings=Settings(path=tmp_path / "s.json"), prompter=_allow
    )
    await resumed_storage.flush()

    assert result.reason is StopReason.COMPLETED
    assert target.read_text() == "hello there"


async def test_resume_repairs_dangling_tool_call(tmp_path):
    """A session crashed mid-tool resumes API-valid (dangling call repaired)."""
    path = session_file(str(tmp_path), "sid", root=tmp_path / "root")
    storage = SessionStorage(path, "sid")
    storage.append_message({"role": "user", "content": "read it"})
    storage.append_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        }
    )
    await storage.flush()  # crash before tool result is recorded

    resumed = load_session(path)
    # Every assistant tool_call is now matched by a tool message.
    call_ids = {
        tc["id"]
        for m in resumed
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    result_ids = {m["tool_call_id"] for m in resumed if m.get("role") == "tool"}
    assert call_ids == result_ids
