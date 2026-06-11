"""Wire-level extensibility checks through the ADK driver (Phases 6-7).

The hook/skill/plugin/memory suites already run through the driver; what they
can't see is the provider payload. These tests capture the ``tools`` kwarg
litellm receives and assert schema advertisement is lossless for both
Pydantic built-ins and raw-JSON-Schema MCP tools, and that the MCP "ask"
permission gate prompts through the full stack.
"""

from __future__ import annotations

import litellm
import pytest

from nano_claude.adk.driver import run_turn
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.extensibility.mcp.tool import MCPTool
from nano_claude.permissions.manager import PromptOutcome
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from nano_claude.tools.registry import clear_dynamic_tools, register_tools
from tests.conftest import FakeStream, text_chunk, tool_call_chunk, usage_chunk


class _Spec:
    def __init__(self, name: str, schema: dict):
        self.name = name
        self.description = f"{name} (mcp)"
        self.inputSchema = schema


class _FakeSession:
    async def call_tool(self, name, args):
        class _Result:
            isError = False
            content = [type("T", (), {"type": "text", "text": f"mcp ran {name} {args}"})()]

        return _Result()


_RAW_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "depth": {"type": "integer", "minimum": 0}},
    "required": ["path"],
    "additionalProperties": False,
}


@pytest.fixture
def mcp_tool():
    tool = MCPTool("filesystem", _FakeSession(), _Spec("list_dir", _RAW_SCHEMA))
    register_tools([tool])
    yield tool
    clear_dynamic_tools()


async def test_tools_payload_advertises_schemas_verbatim(tmp_path, monkeypatch, mcp_tool):
    captured: dict = {}

    async def capturing_acompletion(*args, **kwargs):
        captured["tools"] = kwargs.get("tools")
        return FakeStream([text_chunk("hi"), usage_chunk(3, 1)])

    monkeypatch.setattr(litellm, "acompletion", capturing_acompletion)

    state = LoopState(messages=[{"role": "user", "content": "hello"}])
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS)
    await run_turn(state, config, settings=Settings(path=tmp_path / "s.json"))

    by_name = {t["function"]["name"]: t["function"] for t in captured["tools"]}

    # The MCP tool's raw JSON Schema reaches the provider payload verbatim.
    assert by_name["mcp__filesystem__list_dir"]["parameters"] == _RAW_SCHEMA

    # A Pydantic built-in advertises exactly its model_json_schema().
    from nano_claude.tools.bash import BashTool

    bash = BashTool()
    assert by_name["Bash"]["parameters"] == bash.input_schema.model_json_schema()
    assert by_name["Bash"]["description"] == bash.description


async def test_mcp_tool_ask_gate_prompts_through_driver(tmp_path, monkeypatch, mcp_tool):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        _sequential(
            [
                [
                    tool_call_chunk(0, "m1", "mcp__filesystem__list_dir", '{"path": "/x"}'),
                    usage_chunk(5, 2),
                ],
                [text_chunk("listed"), usage_chunk(6, 1)],
            ]
        ),
    )

    prompts = []

    async def prompter(tool, args, text):
        prompts.append(tool.name)
        return PromptOutcome.ALLOW_ONCE

    state = LoopState(messages=[{"role": "user", "content": "list /x"}])
    # DEFAULT mode: MCP tools always ask — external code is never silently run.
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.DEFAULT)
    result = await run_turn(
        state, config, settings=Settings(path=tmp_path / "s.json"), prompter=prompter
    )

    assert result.reason is StopReason.COMPLETED
    assert prompts == ["mcp__filesystem__list_dir"]
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert tool_msg["content"] == "mcp ran list_dir {'path': '/x'}"


async def test_mcp_tool_denied_when_prompt_refused(tmp_path, monkeypatch, mcp_tool):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        _sequential(
            [
                [
                    tool_call_chunk(0, "m1", "mcp__filesystem__list_dir", '{"path": "/x"}'),
                    usage_chunk(5, 2),
                ],
                [text_chunk("ok, skipped"), usage_chunk(6, 1)],
            ]
        ),
    )

    async def prompter(tool, args, text):
        return PromptOutcome.DENY_ONCE

    state = LoopState(messages=[{"role": "user", "content": "list /x"}])
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.DEFAULT)
    result = await run_turn(
        state, config, settings=Settings(path=tmp_path / "s.json"), prompter=prompter
    )

    assert result.reason is StopReason.COMPLETED
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert tool_msg["content"].startswith("Permission denied")


def _sequential(streams):
    it = iter([FakeStream(s) for s in streams])

    async def _acompletion(*args, **kwargs):
        return next(it)

    return _acompletion
