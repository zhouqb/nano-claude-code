"""Tool registry: the canonical set of built-in tools and lookup helpers."""

from __future__ import annotations

from nano_claude.permissions.modes import PermissionMode
from nano_claude.tools.base import Tool
from nano_claude.tools.bash import BashTool
from nano_claude.tools.edit import EditTool
from nano_claude.tools.glob_tool import GlobTool
from nano_claude.tools.grep import GrepTool
from nano_claude.tools.read import ReadTool
from nano_claude.tools.write import WriteTool

BASE_TOOLS: list[Tool] = [
    BashTool(),
    ReadTool(),
    WriteTool(),
    EditTool(),
    GlobTool(),
    GrepTool(),
]

# MCP- and plugin-contributed tools, registered at startup.
_DYNAMIC_TOOLS: list[Tool] = []


def register_tools(tools: list[Tool]) -> None:
    """Append externally-provided tools (MCP servers, plugins) to the registry."""
    _DYNAMIC_TOOLS.extend(tools)


def clear_dynamic_tools() -> None:
    _DYNAMIC_TOOLS.clear()


def get_tools(permission_mode: PermissionMode) -> list[Tool]:
    """Return the tools available for a given mode.

    The same set is exposed regardless of mode; dynamic (MCP/plugin) tools are
    appended so they are indistinguishable from built-ins downstream.
    """
    return BASE_TOOLS + _DYNAMIC_TOOLS


def get_tool(name: str) -> Tool | None:
    return next((t for t in BASE_TOOLS + _DYNAMIC_TOOLS if t.name == name), None)
