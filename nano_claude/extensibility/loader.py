"""The single startup entry point that wires every extension mechanism.

``load_extensions`` is called once, inside the event loop and before the first
turn, to register settings-configured hooks, discover skills, connect MCP
servers, and load plugins (which contribute more of all three). It must run in
the event loop because MCP connections are async — and in the same task that
later calls :func:`close_mcp`, so the SDK's cancel scopes stay put.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nano_claude.extensibility.hooks import get_hooks, register_hooks
from nano_claude.extensibility.mcp import load_mcp_servers
from nano_claude.extensibility.plugins import load_plugins
from nano_claude.extensibility.skills import load_skills
from nano_claude.permissions.settings import Settings
from nano_claude.tools.registry import register_tools


@dataclass
class ExtensionsSummary:
    """Counts of what was wired, for a one-line startup notice."""

    hooks: int = 0
    skills: int = 0
    mcp_tools: int = 0
    plugins: int = 0

    @property
    def anything(self) -> bool:
        return bool(self.hooks or self.skills or self.mcp_tools or self.plugins)


async def load_extensions(
    settings: Settings,
    *,
    skills_dir: Path | None = None,
    plugins_dir: Path | None = None,
) -> ExtensionsSummary:
    """Wire hooks, skills, MCP servers, and plugins. Returns what was loaded."""
    register_hooks(settings.hooks)
    skills = load_skills(skills_dir)

    mcp_tools = await load_mcp_servers(settings.mcp_servers)
    register_tools(mcp_tools)

    plugins = await load_plugins(plugins_dir)

    # Plugins register their own hooks/skills/tools into the same registries;
    # count their contributions on top of the directly-configured ones.
    plugin_skills = sum(len(p.skills) for p in plugins)
    plugin_mcp = sum(len(p.mcp_tools) for p in plugins)
    return ExtensionsSummary(
        hooks=len(get_hooks()),
        skills=len(skills) + plugin_skills,
        mcp_tools=len(mcp_tools) + plugin_mcp,
        plugins=len(plugins),
    )
