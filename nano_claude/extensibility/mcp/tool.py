"""Wrap an MCP-provided tool as a nano ``Tool``.

An ``MCPTool`` is namespaced ``mcp__<server>__<tool>`` so it is unique across
servers and indistinguishable from a built-in at every later stage (schema,
permissions, hooks). Unlike built-ins it carries the server's raw JSON Schema
(not a Pydantic model) and receives ``dict`` args, so it sets
``reads_raw_args`` and overrides ``to_api_schema`` to emit the schema verbatim.
"""

from __future__ import annotations

import re
from typing import Any

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult

# Claude.ai server names carry this prefix; mirrors Claude Code's normalization.
_CLAUDEAI_SERVER_PREFIX = "claude.ai "
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    """Make a server/tool name fit the API pattern ``^[a-zA-Z0-9_-]{1,64}$``.

    Mirrors Claude Code's ``normalizeNameForMCP``: invalid characters (spaces,
    dots, slashes, …) become underscores. For claude.ai servers, repeated and
    edge underscores are also collapsed so they don't clash with the ``__``
    delimiter in the qualified tool name.
    """
    normalized = _INVALID_NAME_CHARS.sub("_", name)
    if name.startswith(_CLAUDEAI_SERVER_PREFIX):
        normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


class MCPTool(Tool):
    reads_raw_args = True

    def __init__(self, server_name: str, session: Any, spec: Any) -> None:
        # Advertised/permission name is normalized (must be a valid function
        # name and align with mcp__<server>__<tool> rules); the raw tool name is
        # kept for the actual call_tool.
        self.name = f"mcp__{normalize_mcp_name(server_name)}__{normalize_mcp_name(spec.name)}"
        self.description = spec.description or ""
        self._session = session
        self._raw_name = spec.name
        self._json_schema = spec.inputSchema or {"type": "object", "properties": {}}

    def to_api_schema(self) -> dict[str, Any]:
        # Pass the server's schema straight through (no Pydantic round-trip).
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._json_schema,
            },
        }

    async def call(self, args: dict, context: ToolContext) -> ToolResult:
        result = await self._session.call_tool(self._raw_name, args)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return ToolResult(output=text, is_error=bool(result.isError))

    async def check_permissions(self, args: dict, context: ToolContext) -> PermissionDecision:
        # External code is never silently allowed; the user gates each MCP tool.
        return PermissionDecision(behavior="ask", prompt=f"Run MCP tool {self.name}?")
