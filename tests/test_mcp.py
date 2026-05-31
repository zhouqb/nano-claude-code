"""Tests for the MCP client subsystem.

Unit tests use a fake session (no subprocess); the integration test spins up a
real stdio MCP server (FastMCP) and drives it through ``connect_server`` and the
permission path.
"""

from __future__ import annotations

import sys
import textwrap
from types import SimpleNamespace

from nano_claude.extensibility.mcp import (
    HTTPServerConfig,
    MCPTool,
    StdioServerConfig,
    parse_server_config,
)
from nano_claude.permissions.manager import PromptOutcome, has_permission_to_use_tool
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings
from nano_claude.tools.base import ToolContext

# --- config parsing ---------------------------------------------------------


def test_parse_stdio_config():
    cfg = parse_server_config({"command": "npx", "args": ["-y", "srv"]})
    assert isinstance(cfg, StdioServerConfig)
    assert cfg.transport == "stdio"
    assert cfg.args == ["-y", "srv"]


def test_parse_http_config():
    cfg = parse_server_config({"transport": "http", "url": "https://x/mcp"})
    assert isinstance(cfg, HTTPServerConfig)
    assert cfg.url == "https://x/mcp"


def test_settings_parses_mcp_servers(tmp_path):
    import json

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"mcpServers": {"fs": {"command": "npx"}}}))
    settings = Settings.load(path)
    assert settings.mcp_servers == {"fs": {"command": "npx"}}


# --- MCPTool wrapping (fake session) ----------------------------------------


def _spec(name, schema=None):
    return SimpleNamespace(
        name=name, description=f"{name} tool", inputSchema=schema or {"type": "object"}
    )


def test_namespacing_and_schema_passthrough():
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    tool = MCPTool("filesystem", session=None, spec=_spec("read_file", schema))
    assert tool.name == "mcp__filesystem__read_file"
    assert tool.reads_raw_args is True
    api = tool.to_api_schema()
    assert api["function"]["name"] == "mcp__filesystem__read_file"
    assert api["function"]["parameters"] == schema


def test_name_is_normalized_but_raw_name_preserved():
    # Spaces / dots / slashes in server and tool names would make an invalid
    # OpenAI function name; they're normalized for the advertised/permission
    # name while call_tool still uses the raw name.
    tool = MCPTool("my server.v2", session=None, spec=_spec("read/file"))
    assert tool.name == "mcp__my_server_v2__read_file"
    assert tool._raw_name == "read/file"
    # The advertised name matches the API function-name pattern.
    import re

    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tool.name)


def test_claudeai_server_underscores_collapsed():
    tool = MCPTool("claude.ai Slack", session=None, spec=_spec("send"))
    # "claude.ai Slack" → "claude_ai_Slack" (collapsed, no leading/trailing _).
    assert tool.name == "mcp__claude_ai_Slack__send"


async def test_call_joins_text_content():
    class FakeSession:
        async def call_tool(self, name, args):
            assert name == "echo"
            assert args == {"msg": "hi"}
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="text", text="line1"),
                    SimpleNamespace(type="image", text="ignored"),
                    SimpleNamespace(type="text", text="line2"),
                ],
                isError=False,
            )

    tool = MCPTool("srv", FakeSession(), _spec("echo"))
    ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)
    result = await tool.call({"msg": "hi"}, ctx)
    assert result.output == "line1\nline2"
    assert result.is_error is False


async def test_call_propagates_error_flag():
    class FakeSession:
        async def call_tool(self, name, args):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="boom")], isError=True
            )

    tool = MCPTool("srv", FakeSession(), _spec("x"))
    ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)
    result = await tool.call({}, ctx)
    assert result.is_error is True


# --- permission path (dict args) --------------------------------------------


async def test_mcp_tool_asks_by_default():
    tool = MCPTool("srv", session=None, spec=_spec("danger"))
    ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)

    prompted = {}

    async def prompter(t, args, text):
        prompted["text"] = text
        return PromptOutcome.ALLOW_ONCE

    decision = await has_permission_to_use_tool(tool, {"x": 1}, ctx, Settings(), prompter)
    assert decision.behavior == "allow"
    assert "mcp__srv__danger" in prompted["text"]


async def test_mcp_tool_allow_rule_matches_namespaced_name():
    from nano_claude.permissions.rules import PermissionRule

    tool = MCPTool("fs", session=None, spec=_spec("read_file"))
    ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)
    settings = Settings(allow_rules=[PermissionRule("mcp__fs__*", "allow")])

    async def _never(t, a, x):
        raise AssertionError("should not prompt — allow rule matches")

    decision = await has_permission_to_use_tool(tool, {}, ctx, settings, _never)
    assert decision.behavior == "allow"


# --- real stdio server integration ------------------------------------------

_SERVER_SCRIPT = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("calc")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        '''Add two numbers.'''
        return a + b

    if __name__ == "__main__":
        mcp.run()
    """
)


async def test_connect_real_stdio_server_and_call(tmp_path):
    from nano_claude.extensibility.mcp import client as mcp_client

    server = tmp_path / "calc_server.py"
    server.write_text(_SERVER_SCRIPT)

    cfg = StdioServerConfig(command=sys.executable, args=[str(server)])
    try:
        tools = await mcp_client.connect_server("calc", cfg)
        names = {t.name for t in tools}
        assert "mcp__calc__add" in names

        add = next(t for t in tools if t.name == "mcp__calc__add")
        ctx = ToolContext(cwd="/", cancel_event=None, permission_mode=PermissionMode.DEFAULT)
        result = await add.call({"a": 2, "b": 3}, ctx)
        assert "5" in result.output
        assert result.is_error is False
    finally:
        await mcp_client.close_mcp()
