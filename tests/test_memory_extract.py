"""Tests for the turn-end extraction agent: the permission gate + lifecycle."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import litellm

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.memory.extract import (
    ExtractionManager,
    create_memory_can_use_tool,
    has_memory_writes_since,
)
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from tests.conftest import make_sequential_acompletion, text_chunk, tool_call_chunk, usage_chunk


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


# --- permission gate --------------------------------------------------------


async def test_gate_allows_read_tools(tmp_path):
    gate = create_memory_can_use_tool(tmp_path)
    for name in ("Read", "Grep", "GlobTool"):
        assert (await gate(_tool(name), {}, None)).behavior == "allow"


async def test_gate_allows_write_inside_mdir(tmp_path):
    gate = create_memory_can_use_tool(tmp_path)
    args = {"file_path": str(tmp_path / "user.md")}
    assert (await gate(_tool("Write"), args, None)).behavior == "allow"
    assert (await gate(_tool("Edit"), args, None)).behavior == "allow"


async def test_gate_denies_write_outside_mdir(tmp_path):
    gate = create_memory_can_use_tool(tmp_path)
    decision = await gate(_tool("Write"), {"file_path": str(tmp_path.parent / "escape.md")}, None)
    assert decision.behavior == "deny"


async def test_gate_denies_other_tools(tmp_path):
    gate = create_memory_can_use_tool(tmp_path)
    for name in ("Bash", "Task", "mcp__x__y"):
        assert (await gate(_tool(name), {}, None)).behavior == "deny"


# --- has_memory_writes_since ------------------------------------------------


def _assistant_write(path: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c", "function": {"name": "Write", "arguments": json.dumps({"file_path": path})}}
        ],
    }


def test_detects_main_agent_write_to_mdir(tmp_path):
    messages = [{"role": "user", "content": "hi"}, _assistant_write(str(tmp_path / "a.md"))]
    assert has_memory_writes_since(messages, 0, tmp_path) is True


def test_ignores_writes_outside_mdir(tmp_path):
    messages = [_assistant_write(str(tmp_path.parent / "other.md"))]
    assert has_memory_writes_since(messages, 0, tmp_path) is False


def test_respects_since_index(tmp_path):
    messages = [_assistant_write(str(tmp_path / "a.md")), {"role": "user", "content": "next"}]
    # Starting after the write, there are no new memory writes.
    assert has_memory_writes_since(messages, 1, tmp_path) is False


# --- permission_override is honored by the loop -----------------------------


async def test_loop_override_blocks_write_outside_mdir(tmp_path, monkeypatch):
    mdir = tmp_path / "mem"
    mdir.mkdir()
    target = tmp_path / "outside.txt"  # deliberately outside the memory dir
    write_args = json.dumps({"file_path": str(target), "content": "nope"})
    streams = [
        [tool_call_chunk(0, "c1", "Write", write_args), usage_chunk(5, 5)],
        [text_chunk("ok"), usage_chunk(1, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "go"}])
    # Even under BYPASS, the override is the whole policy and confines writes.
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS)
    result = await query_loop(
        state,
        config,
        settings=Settings(),
        allowed_tools=["Write"],
        permission_override=create_memory_can_use_tool(mdir),
    )

    assert result.reason is StopReason.COMPLETED
    assert not target.exists()
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "memory directory" in tool_msg["content"]


# --- ExtractionManager lifecycle --------------------------------------------


async def test_schedule_skips_when_main_agent_already_wrote(tmp_path, monkeypatch):
    mdir = tmp_path / "mem"
    mdir.mkdir()
    state = LoopState(
        messages=[{"role": "user", "content": "hi"}, _assistant_write(str(mdir / "a.md"))]
    )
    mgr = ExtractionManager(state, AgentConfig(cwd=str(tmp_path)), Settings(), mdir)

    ran = False

    async def _fake_run(end):
        nonlocal ran
        ran = True

    monkeypatch.setattr(mgr, "_run", _fake_run)
    mgr.schedule()
    if mgr._task:
        await mgr._task

    assert ran is False  # main agent already saved → fork skipped
    assert mgr.cursor == len(state.messages)  # cursor advanced past the range


async def test_schedule_runs_and_advances_cursor(tmp_path, monkeypatch):
    mdir = tmp_path / "mem"
    mdir.mkdir()
    state = LoopState(messages=[{"role": "user", "content": "remember I like uv"}])
    mgr = ExtractionManager(state, AgentConfig(cwd=str(tmp_path)), Settings(), mdir)

    calls: list[int] = []

    async def _fake_run(end):
        calls.append(end)

    monkeypatch.setattr(mgr, "_run", _fake_run)
    mgr.schedule()
    assert mgr._task is not None
    await mgr._task

    assert calls == [1]
    assert mgr.cursor == 1
    # Nothing new since the cursor → a second schedule is a no-op.
    mgr.schedule()
    assert mgr._task.done()


async def test_drain_awaits_inflight(tmp_path, monkeypatch):
    mdir = tmp_path / "mem"
    mdir.mkdir()
    state = LoopState(messages=[{"role": "user", "content": "x"}])
    mgr = ExtractionManager(state, AgentConfig(cwd=str(tmp_path)), Settings(), mdir)

    finished = False

    async def _slow_run(end):
        nonlocal finished
        await asyncio.sleep(0.05)
        finished = True

    monkeypatch.setattr(mgr, "_run", _slow_run)
    mgr.schedule()
    await mgr.drain(timeout=5)
    assert finished is True
