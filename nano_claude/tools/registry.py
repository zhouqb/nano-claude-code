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


def get_tools(permission_mode: PermissionMode) -> list[Tool]:
    """Return the tools available for a given mode.

    Phase 2 exposes the same set regardless of mode; plugin-contributed tools
    will be merged here in a later phase.
    """
    return BASE_TOOLS


def get_tool(name: str) -> Tool | None:
    return next((t for t in BASE_TOOLS if t.name == name), None)
