"""Tests for the lifecycle hooks subsystem."""

from __future__ import annotations

import json

import litellm
import pytest

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.extensibility.hooks import (
    HookDefinition,
    HookEvent,
    clear_hooks,
    execute_hooks,
    register_hooks,
)
from nano_claude.permissions.manager import PromptOutcome
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from tests.conftest import make_sequential_acompletion, text_chunk, tool_call_chunk, usage_chunk


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty global hook registry."""
    clear_hooks()
    yield
    clear_hooks()


def _hook(event: str, command: str, **kw) -> HookDefinition:
    return HookDefinition(event=event, command=command, **kw)


# --- payload + basic execution ---------------------------------------------


async def test_post_tool_use_stdout_becomes_context():
    register_hooks([_hook("PostToolUse", "cat")])  # echoes the stdin payload
    outcome = await execute_hooks(
        HookEvent.POST_TOOL_USE,
        session_id="sess1",
        cwd="/proj",
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response="file.txt",
    )
    assert not outcome.blocked
    payload = json.loads(outcome.context_text)
    assert payload["hook_event_name"] == "PostToolUse"
    assert payload["session_id"] == "sess1"
    assert payload["cwd"] == "/proj"
    assert payload["tool_name"] == "Bash"
    assert payload["tool_input"] == {"command": "ls"}
    assert payload["tool_response"] == "file.txt"


async def test_pre_tool_use_payload_omits_response():
    register_hooks([_hook("PreToolUse", "cat")])
    # cat exits 0; for PreToolUse stdout is still collected as context.
    outcome = await execute_hooks(
        HookEvent.PRE_TOOL_USE,
        tool_name="Read",
        tool_input={"file_path": "/x"},
    )
    payload = json.loads(outcome.context_text)
    assert "tool_response" not in payload
    assert payload["tool_name"] == "Read"


# --- blocking semantics -----------------------------------------------------


async def test_exit_2_blocks_pre_tool_use():
    register_hooks([_hook("PreToolUse", ">&2 echo 'no rm allowed'; exit 2")])
    outcome = await execute_hooks(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={})
    assert outcome.blocked
    assert "no rm allowed" in outcome.block_reason


async def test_exit_2_default_reason_when_silent():
    register_hooks([_hook("PreToolUse", "exit 2")])
    outcome = await execute_hooks(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={})
    assert outcome.blocked
    assert outcome.block_reason == "Blocked by hook"


async def test_first_deny_short_circuits():
    register_hooks(
        [
            _hook("PreToolUse", "exit 2"),
            _hook("PreToolUse", "echo should-not-run"),
        ]
    )
    outcome = await execute_hooks(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={})
    assert outcome.blocked
    assert "should-not-run" not in outcome.context_text


async def test_exit_2_on_post_tool_use_is_feedback_not_block():
    register_hooks([_hook("PostToolUse", ">&2 echo 'lint failed'; exit 2")])
    outcome = await execute_hooks(
        HookEvent.POST_TOOL_USE, tool_name="Edit", tool_input={}, tool_response="ok"
    )
    assert not outcome.blocked
    assert "lint failed" in outcome.context_text


async def test_other_nonzero_is_warning_not_block():
    register_hooks([_hook("PreToolUse", ">&2 echo oops; exit 1")])
    outcome = await execute_hooks(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={})
    assert not outcome.blocked
    assert any("oops" in w for w in outcome.warnings)


# --- matchers ---------------------------------------------------------------


async def test_matcher_filters_by_tool_and_arg():
    register_hooks([_hook("PreToolUse", "echo ran", matcher="Bash(git *)")])

    matched = await execute_hooks(
        HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={"command": "git status"}
    )
    assert "ran" in matched.context_text

    skipped = await execute_hooks(
        HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={"command": "npm test"}
    )
    assert "ran" not in skipped.context_text


async def test_only_matching_event_fires():
    register_hooks([_hook("Stop", "echo stop-ran")])
    outcome = await execute_hooks(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={})
    assert outcome.context_text == ""


# --- timeout ----------------------------------------------------------------


async def test_timeout_is_a_warning_not_a_block():
    register_hooks([_hook("PreToolUse", "sleep 5", timeout=0.1)])
    outcome = await execute_hooks(HookEvent.PRE_TOOL_USE, tool_name="Bash", tool_input={})
    assert not outcome.blocked
    assert any("timed out" in w for w in outcome.warnings)


# --- settings parsing -------------------------------------------------------


# --- loop integration -------------------------------------------------------


async def _allow_once(tool, args, text):
    return PromptOutcome.ALLOW_ONCE


async def test_pre_tool_use_hook_blocks_tool_in_loop(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("hello world")
    register_hooks([_hook("PreToolUse", ">&2 echo 'edits frozen'; exit 2", matcher="Edit")])

    edit_args = json.dumps({"file_path": str(target), "old_string": "world", "new_string": "there"})
    streams = [
        [tool_call_chunk(0, "call_1", "Edit", edit_args), usage_chunk(10, 5)],
        [text_chunk("ok"), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "edit it"}])
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS)
    result = await query_loop(state, config, settings=Settings(), prompter=_allow_once)

    assert result.reason is StopReason.COMPLETED
    assert target.read_text() == "hello world"  # tool never ran
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "Blocked by hook" in tool_msg["content"]
    assert "edits frozen" in tool_msg["content"]


async def test_post_tool_use_hook_appends_context_in_loop(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("data")
    register_hooks([_hook("PostToolUse", "echo formatted", matcher="Read")])

    read_args = json.dumps({"file_path": str(target)})
    streams = [
        [tool_call_chunk(0, "call_1", "Read", read_args), usage_chunk(10, 5)],
        [text_chunk("done"), usage_chunk(2, 1)],
    ]
    monkeypatch.setattr(litellm, "acompletion", make_sequential_acompletion(streams))

    state = LoopState(messages=[{"role": "user", "content": "read it"}])
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS)
    result = await query_loop(state, config, settings=Settings(), prompter=_allow_once)

    assert result.reason is StopReason.COMPLETED
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert "[hook] formatted" in tool_msg["content"]


# --- settings parsing -------------------------------------------------------


def test_settings_parses_hooks(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "hooks": [
                    {"event": "PreToolUse", "matcher": "Bash(rm *)", "command": "block.sh"},
                    {"event": "PostToolUse", "command": "fmt.sh", "async": True, "timeout": 5},
                    {"event": "NotAnEvent", "command": "skip.sh"},  # invalid → skipped
                ]
            }
        )
    )
    settings = Settings.load(path)
    assert len(settings.hooks) == 2
    assert settings.hooks[0].matcher == "Bash(rm *)"
    assert settings.hooks[1].run_async is True
    assert settings.hooks[1].timeout_s == 5
