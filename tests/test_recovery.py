"""Tests for session restore and crash recovery."""

from __future__ import annotations

from nano_claude.session.restore import (
    INTERRUPTED,
    last_assistant_ts,
    list_sessions,
    load_records,
    load_session,
    repair_messages,
    restore_messages,
)
from nano_claude.session.storage import MessageRecord, SessionStorage, session_file


def _msg(uuid: str, message: dict, ts: float = 1.0) -> MessageRecord:
    return MessageRecord(uuid=uuid, ts=ts, message=message)


# --- last_assistant_ts (microcompact time-gate seed on resume) --------------


def test_last_assistant_ts_returns_most_recent():
    records = [
        _msg("a", {"role": "user", "content": "hi"}, ts=10.0),
        _msg("b", {"role": "assistant", "content": "first"}, ts=20.0),
        _msg("c", {"role": "user", "content": "more"}, ts=30.0),
        _msg("d", {"role": "assistant", "content": "second"}, ts=40.0),
    ]
    assert last_assistant_ts(records) == 40.0


def test_last_assistant_ts_none_without_assistant():
    records = [_msg("a", {"role": "user", "content": "hi"}, ts=10.0)]
    assert last_assistant_ts(records) is None


# --- restore_messages -------------------------------------------------------


def test_restore_dedups_by_uuid():
    records = [
        _msg("a", {"role": "user", "content": "hi"}),
        _msg("a", {"role": "user", "content": "hi"}),  # duplicate uuid
        _msg("b", {"role": "assistant", "content": "yo"}),
    ]
    messages = restore_messages(records)
    assert len(messages) == 2


# --- repair_messages --------------------------------------------------------


def test_repair_injects_missing_tool_result():
    messages = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        # crashed here: no tool result for call_1
    ]
    repaired = repair_messages(messages)
    tool_msgs = [m for m in repaired if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    assert tool_msgs[0]["content"] == INTERRUPTED


def test_repair_leaves_resolved_calls_untouched():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]
    assert repair_messages(messages) == messages


def test_repair_handles_partial_multi_call():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    repaired = repair_messages(messages)
    resolved = {m["tool_call_id"] for m in repaired if m.get("role") == "tool"}
    assert resolved == {"c1", "c2"}


def test_repair_noop_for_plain_conversation():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert repair_messages(messages) == messages


# --- end-to-end: simulate a mid-turn crash ----------------------------------


async def test_mid_turn_crash_then_resume(tmp_path):
    path = session_file("/proj", "sid", root=tmp_path)
    storage = SessionStorage(path, "sid")
    storage.append_metadata(model="m", cwd="/proj")
    storage.append_message({"role": "user", "content": "read the file"})
    storage.append_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{}"},
                }
            ],
        }
    )
    await storage.flush()  # process "dies" here, before the tool result

    # Resume from disk.
    messages = load_session(path)
    # Critical invariant holds: the dangling tool_call now has a tool result.
    assert messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": INTERRUPTED}
    user_msgs = [m for m in messages if m.get("role") == "user"]
    assert user_msgs[0]["content"] == "read the file"


# --- list_sessions ----------------------------------------------------------


async def test_list_sessions(tmp_path):
    path = session_file("/proj", "sid", root=tmp_path)
    storage = SessionStorage(path, "sid")
    storage.append_metadata(model="gpt-4o", cwd="/proj")
    storage.append_message({"role": "user", "content": "do the thing"})
    await storage.flush()

    sessions = list_sessions("/proj", root=tmp_path)
    assert len(sessions) == 1
    assert sessions[0].session_id == "sid"
    assert sessions[0].model == "gpt-4o"
    assert "do the thing" in sessions[0].preview


def test_list_sessions_empty(tmp_path):
    assert list_sessions("/nonexistent", root=tmp_path) == []


def test_load_records_missing_file(tmp_path):
    assert load_records(tmp_path / "nope.jsonl") == []
