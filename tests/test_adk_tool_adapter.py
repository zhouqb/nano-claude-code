"""Tests for ``NanoToolAdapter`` — nano ``Tool`` behind ADK's tool protocol.

ADK's ``ToolContext`` is never exercised by the adapter (all nano state rides
in the bound nano ``ToolContext``), so tests pass ``tool_context=None``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import BaseModel

from nano_claude.adk.tool_adapter import NanoToolAdapter
from nano_claude.permissions.modes import PermissionMode
from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult
from nano_claude.tools.registry import BASE_TOOLS


def ctx(cwd: str = "/tmp") -> ToolContext:
    return ToolContext(
        cwd=cwd,
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.DEFAULT,
        output_dir=Path(cwd) / "_overflow",
    )


class EchoInput(BaseModel):
    text: str


class EchoTool(Tool):
    name = "Echo"
    description = "Echo the input back."
    input_schema = EchoInput

    async def call(self, args: EchoInput, context: ToolContext) -> ToolResult:
        return ToolResult(output=f"echo: {args.text}")

    async def check_permissions(self, args, context) -> PermissionDecision:
        return PermissionDecision(behavior="allow")


class ExplodingTool(EchoTool):
    name = "Explode"

    async def call(self, args, context) -> ToolResult:
        raise RuntimeError("boom")


class RawArgsTool(Tool):
    name = "mcp__srv__raw"
    description = "MCP-style raw-schema tool."
    input_schema = EchoInput  # unused: reads_raw_args bypasses it
    reads_raw_args = True
    _json_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    def to_api_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._json_schema,
            },
        }

    async def call(self, args: dict, context: ToolContext) -> ToolResult:
        assert isinstance(args, dict)  # raw dict, no Pydantic round-trip
        return ToolResult(output=f"raw: {args}")

    async def check_permissions(self, args, context) -> PermissionDecision:
        return PermissionDecision(behavior="ask", prompt="?")


# --- declaration fidelity -----------------------------------------------------


def test_declarations_match_to_api_schema_for_all_base_tools():
    context = ctx()
    for tool in BASE_TOOLS:
        decl = NanoToolAdapter(tool, context)._get_declaration()
        fn = tool.to_api_schema()["function"]
        assert decl.name == fn["name"]
        assert decl.description == fn["description"]
        # The exact dict, not a lossy types.Schema round-trip: this is what the
        # pinned ADK LiteLlm converter forwards to the OpenAI tools payload.
        assert decl.parameters_json_schema == fn["parameters"]
        assert decl.parameters is None


def test_mcp_style_raw_schema_passes_through_verbatim():
    decl = NanoToolAdapter(RawArgsTool(), ctx())._get_declaration()
    assert decl.parameters_json_schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }


# --- run_async ----------------------------------------------------------------


async def test_run_async_validates_and_returns_output_string():
    out = await NanoToolAdapter(EchoTool(), ctx()).run_async(args={"text": "hi"}, tool_context=None)
    assert out == "echo: hi"


async def test_run_async_invalid_args_return_error_string():
    out = await NanoToolAdapter(EchoTool(), ctx()).run_async(args={}, tool_context=None)
    assert out.startswith("Error: invalid arguments for Echo:")


async def test_run_async_raw_args_skip_validation():
    out = await NanoToolAdapter(RawArgsTool(), ctx()).run_async(
        args={"q": "x", "extra": 1}, tool_context=None
    )
    assert out == "raw: {'q': 'x', 'extra': 1}"


async def test_run_async_contains_tool_exceptions():
    out = await NanoToolAdapter(ExplodingTool(), ctx()).run_async(
        args={"text": "hi"}, tool_context=None
    )
    assert out == "Tool raised an exception: boom"


async def test_run_async_short_circuits_on_cancel():
    context = ctx()
    context.cancel_event.set()
    out = await NanoToolAdapter(EchoTool(), context).run_async(
        args={"text": "hi"}, tool_context=None
    )
    assert out == "[Interrupted]"


async def test_run_async_fires_display_callbacks():
    events: list[tuple] = []
    adapter = NanoToolAdapter(
        EchoTool(),
        ctx(),
        on_tool_start=lambda name, args: events.append(("start", name, args)),
        on_tool_end=lambda name, result: events.append(("end", name, result.output)),
    )
    await adapter.run_async(args={"text": "hi"}, tool_context=None)
    assert events == [("start", "Echo", {"text": "hi"}), ("end", "Echo", "echo: hi")]
