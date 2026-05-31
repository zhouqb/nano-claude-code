"""Plugin manifest types.

A plugin is a directory under ``~/.nano-claude/plugins/`` with a
``manifest.json`` that bundles the other three extension mechanisms — hooks,
skills, and MCP servers — so installing one directory wires all of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from nano_claude.extensibility.hooks.types import HookDefinition
from nano_claude.extensibility.skills.types import SkillDefinition
from nano_claude.tools.base import Tool


class PluginManifest(BaseModel):
    name: str
    version: str = "0.0.0"
    enabled: bool = True
    hooks: list[HookDefinition] = Field(default_factory=list)
    # Directories (relative to the plugin dir, or absolute) holding skill files.
    skills_paths: list[str] = Field(default_factory=list)
    # Same shape as settings.json "mcpServers".
    mcp_servers: dict[str, dict] = Field(default_factory=dict)


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    hooks: list[HookDefinition] = field(default_factory=list)
    skills: list[SkillDefinition] = field(default_factory=list)
    mcp_tools: list[Tool] = field(default_factory=list)
