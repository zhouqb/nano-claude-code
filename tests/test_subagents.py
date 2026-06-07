"""Tests for Phase 6 subagents: loader, Task tool, runner, isolation."""

from __future__ import annotations

import asyncio
import json

import litellm
import pytest

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason, TokenUsage
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from nano_claude.subagents import (
    AgentDefinition,
    clear_agents,
    get_agent,
    load_agents,
    register_agent,
)
from nano_claude.subagents.runner import _resolve_allowed_tools, run_subagent_loop
from nano_claude.tools.base import ToolContext
from nano_claude.tools.task import TaskTool
from tests.conftest import (
    FakeStream,
    make_sequential_acompletion,
    text_chunk,
    tool_call_chunk,
    usage_chunk,
)


@pytest.fixture(autouse=True)
def _clean_agents():
    clear_agents()
    register_agent(
        AgentDefinition(name="explorer", description="read-only search", system_prompt="explore")
    )
    yield
    clear_agents()


# --- loader -----------------------------------------------------------------


def test_load_agents_registers_builtin(tmp_path):
    clear_agents()
    loaded = load_agents(tmp_path / "nope")
    assert get_agent("general-purpose") is not None
    assert any(a.name == "general-purpose" for a in loaded)


def test_load_agents_registers_verification(tmp_path):
    clear_agents()
    loaded = load_agents(tmp_path / "nope")
    agent = get_agent("verification")
    assert agent is not None
    assert any(a.name == "verification" for a in loaded)
    # adversarial framing and the red→green / VERDICT contract are present
    assert "try to break it" in agent.system_prompt.lower()
    assert "VERDICT:" in agent.system_prompt
    assert "reproduce the original bug" in agent.system_prompt.lower()


def test_verification_agent_is_read_only(tmp_path):
    clear_agents()
    load_agents(tmp_path / "nope")
    agent = get_agent("verification")
    # The verifier may run things and read, but must not be able to edit/write.
    assert "Edit" not in (agent.tools or [])
    assert "Write" not in (agent.tools or [])
    # Even unrestricted-tool resolution always drops Task; here the explicit
    # allow-list must resolve to exactly the read-only subset.
    resolved = _resolve_allowed_tools(agent, PermissionMode.DEFAULT)
    assert "Edit" not in resolved and "Write" not in resolved and "Task" not in resolved
    assert "Bash" in resolved and "Read" in resolved


def test_load_agents_from_markdown(tmp_path):
    clear_agents()
    (tmp_path / "reviewer.md").write_text(
        "---\n"
        "name: reviewer\n"
        "description: Reviews diffs\n"
        "tools: [Read, Grep]\n"
        "model: anthropic/claude-haiku-4-5\n"
        "---\n"
        "You review code. Report issues with file:line refs.\n"
    )
    load_agents(tmp_path)
    agent = get_agent("reviewer")
    assert agent.description == "Reviews diffs"
    assert agent.tools == ["Read", "Grep"]
    assert agent.model == "anthropic/claude-haiku-4-5"
    assert "review code" in agent.system_prompt.lower()


# --- Task tool --------------------------------------------------------------


def test_task_description_enumerates_agents():
    desc = TaskTool().description
    assert "explorer" in desc
    assert "read-only search" in desc


async def test_task_denies_unknown_agent():
    tool = TaskTool()
    ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)
    args = tool.input_schema(subagent_type="ghost", description="x", prompt="p")
    decision = await tool.check_permissions(args, ctx)
    assert decision.behavior == "deny"
    assert "ghost" in decision.reason


async def test_task_allows_known_agent():
    tool = TaskTool()
    ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)
    args = tool.input_schema(subagent_type="explorer", description="x", prompt="p")
    decision = await tool.check_permissions(args, ctx)
    assert decision.behavior == "allow"


# --- tool resolution (recursion cap) ----------------------------------------


def test_resolve_tools_excludes_task_when_unrestricted():
    agent = AgentDefinition(name="a", description="d", system_prompt="s", tools=None)
    names = _resolve_allowed_tools(agent, PermissionMode.DEFAULT)
    assert "Task" not in names
    assert "Read" in names


def test_resolve_tools_filters_to_subset_minus_task():
    agent = AgentDefinition(name="a", description="d", system_prompt="s", tools=["Read", "Task"])
    names = _resolve_allowed_tools(agent, PermissionMode.DEFAULT)
    assert names == ["Read"]  # Task dropped even if explicitly requested


# --- runner isolation -------------------------------------------------------


async def test_run_subagent_returns_only_final_text(monkeypatch):
    # Subagent does one turn of plain text, no tools.
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_sequential_acompletion([[text_chunk("found 3 matches"), usage_chunk(8, 4)]]),
    )
    parent_usage = TokenUsage()
    ctx = ToolContext(
        cwd="/",
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        parent_model="anthropic/claude-sonnet-4-6",
        token_usage_sink=parent_usage,
        settings=Settings(),
    )
    agent = get_agent("explorer")
    result = await run_subagent_loop(agent, "find matches", ctx)

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "found 3 matches"
    # Cost rolled up to the parent.
    assert parent_usage.input_tokens == 8
    assert parent_usage.output_tokens == 4


async def test_parent_transcript_only_sees_task_call_and_summary(monkeypatch):
    task_args = json.dumps(
        {"subagent_type": "explorer", "description": "search", "prompt": "find X"}
    )
    streams = [
        [tool_call_chunk(0, "call_1", "Task", task_args), usage_chunk(10, 5)],  # parent turn 1
        [text_chunk("X is in foo.py:42"), usage_chunk(6, 3)],  # subagent
        [text_chunk("Done — see foo.py:42."), usage_chunk(4, 2)],  # parent turn 2
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "where is X?"}])
    config = AgentConfig(permission_mode=PermissionMode.BYPASS)
    result = await query_loop(state, config, settings=Settings())

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "Done — see foo.py:42."

    # The parent transcript holds the Task call + its tool result, and the
    # subagent's internal text is delivered ONLY as that tool result.
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "X is in foo.py:42"
    # The subagent's system/user messages never entered the parent transcript.
    assert all(m.get("content") != "explore" for m in state.messages)


async def test_concurrent_task_calls_both_return(monkeypatch):
    a1 = json.dumps({"subagent_type": "explorer", "description": "a", "prompt": "p1"})
    a2 = json.dumps({"subagent_type": "explorer", "description": "b", "prompt": "p2"})
    streams = [
        [
            tool_call_chunk(0, "c1", "Task", a1),
            tool_call_chunk(1, "c2", "Task", a2),
            usage_chunk(10, 5),
        ],
        [text_chunk("result one"), usage_chunk(3, 1)],
        [text_chunk("result two"), usage_chunk(3, 1)],
        [text_chunk("both done"), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "do both"}])
    config = AgentConfig(permission_mode=PermissionMode.BYPASS)
    result = await query_loop(state, config, settings=Settings())

    assert result.reason is StopReason.COMPLETED
    tool_results = {m["content"] for m in state.messages if m.get("role") == "tool"}
    assert tool_results == {"result one", "result two"}


async def test_subagent_cannot_call_task(monkeypatch):
    """The subagent loop must not advertise Task (recursion cap)."""
    captured: list = []

    async def _capturing(*args, **kwargs):
        captured.append({t["function"]["name"] for t in (kwargs.get("tools") or [])})
        return FakeStream([text_chunk("ok"), usage_chunk(1, 1)])

    monkeypatch.setattr(litellm, "acompletion", _capturing)

    ctx = ToolContext(
        cwd="/",
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        settings=Settings(),
    )
    await run_subagent_loop(get_agent("explorer"), "go", ctx)

    assert captured  # the subagent made at least one model call
    assert all("Task" not in advertised for advertised in captured)


async def test_subagent_emitting_task_is_refused(monkeypatch):
    """Even if a subagent emits an (unadvertised) Task call, it's refused at
    resolution — no nested subagent is spawned (recursion cap holds)."""
    task_args = json.dumps({"subagent_type": "explorer", "description": "x", "prompt": "recurse"})
    streams = iter(
        [
            [tool_call_chunk(0, "c1", "Task", task_args), usage_chunk(5, 2)],  # tries Task
            [text_chunk("done without recursing"), usage_chunk(2, 1)],
        ]
    )
    calls = {"n": 0, "messages": []}

    async def _acompletion(*args, **kwargs):
        calls["n"] += 1
        calls["messages"].append(kwargs.get("messages"))
        return FakeStream(next(streams))

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    ctx = ToolContext(
        cwd="/",
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.BYPASS,
        settings=Settings(),
    )
    result = await run_subagent_loop(get_agent("explorer"), "go", ctx)

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "done without recursing"
    # Exactly two model calls — a third would mean a nested subagent was spawned.
    assert calls["n"] == 2
    # The refused Task surfaced as a tool result rather than running.
    tool_results = [m for msgs in calls["messages"] for m in msgs if m.get("role") == "tool"]
    assert any("not permitted" in m["content"] for m in tool_results)
