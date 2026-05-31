"""Plugins: directories that bundle hooks, skills, and MCP servers."""

from nano_claude.extensibility.plugins.loader import (
    DEFAULT_PLUGINS_DIR,
    load_plugin,
    load_plugins,
)
from nano_claude.extensibility.plugins.types import LoadedPlugin, PluginManifest

__all__ = [
    "DEFAULT_PLUGINS_DIR",
    "LoadedPlugin",
    "PluginManifest",
    "load_plugin",
    "load_plugins",
]
