"""Wrap an MCP-provided tool as a nano ``Tool``.

An ``MCPTool`` is namespaced ``mcp__<server>__<tool>`` so it is unique across
servers and indistinguishable from a built-in at every later stage (schema,
permissions, hooks). Unlike built-ins it carries the server's raw JSON Schema
(not a Pydantic model) and receives ``dict`` args, so it sets
``reads_raw_args`` and overrides ``to_api_schema`` to emit the schema verbatim.
"""

from __future__ import annotations

from typing import Any

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult


class MCPTool(Tool):
    reads_raw_args = True

    def __init__(self, server_name: str, session: Any, spec: Any) -> None:
        self.name = f"mcp__{server_name}__{spec.name}"
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
