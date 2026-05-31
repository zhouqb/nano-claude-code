"""End-to-end tests spanning multiple subsystems through the real query loop.

These exercise the loop together with session storage, permissions, hooks,
skills, and subagents — the integration seams the per-module tests don't cover.
"""

from __future__ import annotations

import json

import litellm
import pytest

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.extensibility.hooks import HookDefinition, clear_hooks, register_hooks
from nano_claude.extensibility.skills import (
    SkillContext,
    SkillDefinition,
    clear_skills,
    dispatch_skill,
    register_skill,
)
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from nano_claude.session.restore import load_records, repair_messages, restore_messages
from nano_claude.session.storage import SessionStorage, session_file
from nano_claude.subagents import AgentDefinition, clear_agents, register_agent
from tests.conftest import make_sequential_acompletion, text_chunk, tool_call_chunk, usage_chunk


@pytest.fixture(autouse=True)
def _clean_registries():
    clear_hooks()
    clear_skills()
    clear_agents()
    yield
    clear_hooks()
    clear_skills()
    clear_agents()


def _bypass(tmp_path) -> AgentConfig:
    return AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS)


# --- session: persistence + token accounting + transcript validity ----------


async def test_full_session_persists_and_accounts(tmp_path, monkeypatch):
    target = tmp_path / "notes.txt"
    target.write_text("alpha\nbeta")

    read_args = json.dumps({"file_path": str(target)})
    streams = [
        [tool_call_chunk(0, "call_1", "Read", read_args), usage_chunk(20, 8)],
        [text_chunk("The file has two lines."), usage_chunk(5, 3)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    path = session_file(str(tmp_path), "e2e", root=tmp_path)
    storage = SessionStorage(path, "e2e")
    storage.append_metadata(model="m", cwd=str(tmp_path))
    state = LoopState(messages=[{"role": "system", "content": "sys"}], storage=storage)
    user = {"role": "user", "content": "read notes"}
    state.messages.append(user)
    storage.append_message(user)

    result = await query_loop(state, _bypass(tmp_path), settings=Settings())
    await storage.flush()

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "The file has two lines."
    # Token usage accumulates across both turns.
    assert state.token_usage.input_tokens == 25
    assert state.token_usage.output_tokens == 11

    # The persisted transcript reloads, stays API-valid, and holds the tool output.
    reloaded = repair_messages(restore_messages(load_records(path)))
    tool_msgs = [m for m in reloaded if m.get("role") == "tool"]
    assert tool_msgs and "alpha" in tool_msgs[0]["content"]
    resolved = {m["tool_call_id"] for m in reloaded if m.get("role") == "tool"}
    for m in reloaded:
        for tc in m.get("tool_calls", []) or []:
            assert tc["id"] in resolved  # every call has a matching result


# --- skill dispatch + tool restriction + post-tool hook ----------------------


async def test_skill_then_hook_augmented_tool(tmp_path, monkeypatch):
    target = tmp_path / "main.py"
    target.write_text("print('hi')")

    async def _peek_prompt(args: str, ctx: SkillContext) -> str:
        return f"Inspect {args}"

    register_skill(
        SkillDefinition(
            name="peek",
            description="inspect a file",
            allowed_tools=["Read"],
            get_prompt=_peek_prompt,
        )
    )
    register_hooks([HookDefinition(event="PostToolUse", command="echo linted-ok", matcher="Read")])

    dispatch = await dispatch_skill(
        "/peek main.py", SkillContext(cwd=str(tmp_path), session_id="s")
    )
    assert dispatch is not None
    assert dispatch.prompt == "Inspect main.py"
    assert dispatch.allowed_tools == ["Read"]

    read_args = json.dumps({"file_path": str(target)})
    captured: dict = {}

    async def _acompletion(*args, **kwargs):
        # First call: emit a Read tool call. Second: finish.
        captured.setdefault("tools", {t["function"]["name"] for t in (kwargs.get("tools") or [])})
        n = captured.get("n", 0)
        captured["n"] = n + 1
        from tests.conftest import FakeStream

        if n == 0:
            return FakeStream([tool_call_chunk(0, "c1", "Read", read_args), usage_chunk(10, 4)])
        return FakeStream([text_chunk("looks fine"), usage_chunk(3, 2)])

    monkeypatch.setattr(litellm, "acompletion", _acompletion)

    state = LoopState(messages=[{"role": "user", "content": dispatch.prompt}])
    result = await query_loop(
        state, _bypass(tmp_path), settings=Settings(), allowed_tools=dispatch.allowed_tools
    )

    assert result.reason is StopReason.COMPLETED
    # The skill restricted the advertised tools to Read only.
    assert captured["tools"] == {"Read"}
    # The tool result carries both the file content and the post-tool hook output.
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "print('hi')" in tool_msg["content"]
    assert "[hook] linted-ok" in tool_msg["content"]


# --- subagent delegation keeps the parent transcript clean -------------------


async def test_subagent_delegation_end_to_end(tmp_path, monkeypatch):
    register_agent(
        AgentDefinition(name="explorer", description="search", system_prompt="explore quietly")
    )
    task_args = json.dumps(
        {"subagent_type": "explorer", "description": "hunt", "prompt": "find the bug"}
    )
    streams = [
        [tool_call_chunk(0, "call_1", "Task", task_args), usage_chunk(12, 6)],  # parent
        [text_chunk("bug is in parser.py:88"), usage_chunk(7, 4)],  # subagent
        [text_chunk("Fixed — parser.py:88."), usage_chunk(3, 2)],  # parent wrap-up
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "find and fix the bug"}])
    result = await query_loop(state, _bypass(tmp_path), settings=Settings())

    assert result.reason is StopReason.COMPLETED
    assert result.final_text == "Fixed — parser.py:88."
    # Only the subagent's final summary crossed back as the Task result.
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "bug is in parser.py:88"
    # The subagent's own system prompt never leaked into the parent transcript.
    assert all(m.get("content") != "explore quietly" for m in state.messages)
    # Subagent token cost rolled up into the parent total (12+6 + 7+4 + 3+2).
    assert state.token_usage.total == 34
