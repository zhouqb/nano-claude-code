"""Tests for the TodoWrite tool, its state threading, and the staleness nudge."""

from __future__ import annotations

import asyncio

import pytest

from nano_claude.agent.reminders import (
    TODO_TURNS_BETWEEN_REMINDERS,
    TODO_TURNS_SINCE_WRITE,
    build_todo_reminder,
    maybe_todo_reminder,
    todo_turn_counts,
)
from nano_claude.agent.types import LoopState
from nano_claude.permissions.modes import PermissionMode
from nano_claude.tools.base import ToolContext
from nano_claude.tools.registry import get_tool
from nano_claude.tools.todo import TodoWriteTool


def _context(todos: list[dict]) -> ToolContext:
    return ToolContext(
        cwd=".",
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.DEFAULT,
        todos=todos,
    )


def _item(content: str, status: str) -> dict:
    return {"content": content, "status": status, "activeForm": f"{content}-ing"}


def _run(coro):
    return asyncio.run(coro)


# --- the tool ---------------------------------------------------------------


def test_registered_in_base_tools():
    assert get_tool("TodoWrite") is not None


def test_write_replaces_list_in_place():
    store: list[dict] = []
    ctx = _context(store)
    tool = TodoWriteTool()
    args = tool.input_schema.model_validate(
        {"todos": [_item("Write code", "in_progress"), _item("Run tests", "pending")]}
    )

    result = _run(tool.call(args, ctx))

    assert not result.is_error
    assert "modified successfully" in result.output
    # Mutated in place: same object the loop shares with LoopState.
    assert ctx.todos is store
    assert [t["status"] for t in store] == ["in_progress", "pending"]


def test_all_completed_clears_the_list():
    store = [_item("Old", "pending")]
    ctx = _context(store)
    tool = TodoWriteTool()
    args = tool.input_schema.model_validate(
        {"todos": [_item("Write code", "completed"), _item("Run tests", "completed")]}
    )

    _run(tool.call(args, ctx))

    # Everything done → nothing left to track.
    assert store == []


def test_partial_completion_keeps_list():
    ctx = _context([])
    tool = TodoWriteTool()
    args = tool.input_schema.model_validate(
        {"todos": [_item("a", "completed"), _item("b", "in_progress")]}
    )
    _run(tool.call(args, ctx))
    assert len(ctx.todos) == 2


def test_empty_content_rejected():
    tool = TodoWriteTool()
    with pytest.raises(ValueError):
        tool.input_schema.model_validate(
            {"todos": [{"content": "", "status": "pending", "activeForm": "x"}]}
        )


def test_call_without_store_errors():
    ctx = _context([])
    ctx.todos = None
    tool = TodoWriteTool()
    args = tool.input_schema.model_validate({"todos": [_item("a", "pending")]})
    result = _run(tool.call(args, ctx))
    assert result.is_error


def test_permissions_always_allow():
    tool = TodoWriteTool()
    args = tool.input_schema.model_validate({"todos": [_item("a", "pending")]})
    decision = _run(tool.check_permissions(args, _context([])))
    assert decision.behavior == "allow"


# --- the staleness nudge ----------------------------------------------------


def _assistant_turns(n: int) -> list[dict]:
    return [{"role": "assistant", "content": f"turn {i}"} for i in range(n)]


def test_turn_counts_no_history():
    assert todo_turn_counts([]) == (0, 0)


def test_turn_counts_counts_assistant_turns():
    msgs = [{"role": "user", "content": "hi"}, *_assistant_turns(3)]
    since_write, since_reminder = todo_turn_counts(msgs)
    assert since_write == 3
    assert since_reminder == 3


def test_turn_counts_resets_after_todowrite():
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "TodoWrite"}}],
        },
        *_assistant_turns(2),
    ]
    since_write, _ = todo_turn_counts(msgs)
    # The TodoWrite turn itself is not counted; only the 2 turns after it.
    assert since_write == 2


def test_reminder_fires_after_threshold():
    state = LoopState(messages=_assistant_turns(TODO_TURNS_SINCE_WRITE))
    reminder = maybe_todo_reminder(state, allowed_tools=None)
    assert reminder is not None
    assert reminder["role"] == "user"
    assert "hasn't been used recently" in reminder["content"]
    assert "<system-reminder>" in reminder["content"]


def test_reminder_silent_before_threshold():
    state = LoopState(messages=_assistant_turns(TODO_TURNS_SINCE_WRITE - 1))
    assert maybe_todo_reminder(state, allowed_tools=None) is None


def test_reminder_suppressed_when_todowrite_disallowed():
    state = LoopState(messages=_assistant_turns(TODO_TURNS_SINCE_WRITE))
    assert maybe_todo_reminder(state, allowed_tools=["Read"]) is None


def test_reminder_counts_as_reminder_turn():
    # A prior reminder resets the between-reminders counter, so a fresh reminder
    # only fires again after TURNS_BETWEEN_REMINDERS more assistant turns.
    reminder = build_todo_reminder([])
    msgs = _assistant_turns(TODO_TURNS_SINCE_WRITE) + [reminder] + _assistant_turns(1)
    state = LoopState(messages=msgs)
    assert maybe_todo_reminder(state, allowed_tools=None) is None

    msgs2 = (
        _assistant_turns(TODO_TURNS_SINCE_WRITE)
        + [reminder]
        + _assistant_turns(TODO_TURNS_BETWEEN_REMINDERS)
    )
    state2 = LoopState(messages=msgs2)
    assert maybe_todo_reminder(state2, allowed_tools=None) is not None


def test_reminder_includes_current_todos():
    reminder = build_todo_reminder([_item("Ship feature", "in_progress")])
    assert "Ship feature" in reminder["content"]
    assert "[in_progress]" in reminder["content"]
