"""Plugin discovery and loading.

Each ``~/.nano-claude/plugins/<name>/manifest.json`` is validated, and (unless
disabled) its hooks, skills, and MCP servers are merged into the same global
registries the standalone loaders use — so a plugin's contributions are
indistinguishable from directly-configured ones downstream.
"""

from __future__ import annotations

from pathlib import Path

from nano_claude.extensibility.hooks import register_hooks
from nano_claude.extensibility.mcp import load_mcp_servers
from nano_claude.extensibility.plugins.types import LoadedPlugin, PluginManifest
from nano_claude.extensibility.skills import load_skills
from nano_claude.tools.registry import register_tools

DEFAULT_PLUGINS_DIR = Path.home() / ".nano-claude" / "plugins"


async def load_plugin(plugin_dir: Path) -> LoadedPlugin | None:
    """Load one plugin directory. Returns None if there's no valid manifest."""
    manifest_path = plugin_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = PluginManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a malformed manifest must not break startup
        return None
    if not manifest.enabled:
        return None

    register_hooks(manifest.hooks)

    skills = []
    for raw in manifest.skills_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = plugin_dir / path
        skills.extend(load_skills(path))

    mcp_tools = await load_mcp_servers(manifest.mcp_servers)
    register_tools(mcp_tools)

    return LoadedPlugin(manifest=manifest, hooks=manifest.hooks, skills=skills, mcp_tools=mcp_tools)


async def load_plugins(directory: Path | None = None) -> list[LoadedPlugin]:
    """Discover and load every enabled plugin under ``directory``."""
    directory = directory or DEFAULT_PLUGINS_DIR
    if not directory.is_dir():
        return []
    loaded: list[LoadedPlugin] = []
    for plugin_dir in sorted(directory.iterdir()):
        if not plugin_dir.is_dir():
            continue
        plugin = await load_plugin(plugin_dir)
        if plugin is not None:
            loaded.append(plugin)
    return loaded
