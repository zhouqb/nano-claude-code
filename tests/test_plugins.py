"""Tests for plugins and the load_extensions orchestrator."""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from nano_claude.extensibility.hooks import clear_hooks, get_hooks
from nano_claude.extensibility.loader import load_extensions
from nano_claude.extensibility.mcp import client as mcp_client
from nano_claude.extensibility.plugins import load_plugins
from nano_claude.extensibility.skills import clear_skills, get_skill
from nano_claude.permissions.settings import Settings
from nano_claude.tools.registry import clear_dynamic_tools, get_tool


@pytest.fixture(autouse=True)
def _clean_registries():
    clear_hooks()
    clear_skills()
    clear_dynamic_tools()
    yield
    clear_hooks()
    clear_skills()
    clear_dynamic_tools()


def _make_plugin(root, name, manifest: dict, skills: dict[str, str] | None = None):
    """Create a plugin dir with a manifest and optional skill files."""
    plugin_dir = root / name
    plugin_dir.mkdir()
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest))
    if skills:
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        for fname, text in skills.items():
            (skills_dir / fname).write_text(text)
    return plugin_dir


# --- plugin loading ---------------------------------------------------------


async def test_plugin_contributes_hook_and_skill(tmp_path):
    _make_plugin(
        tmp_path,
        "git-tools",
        {
            "name": "git-tools",
            "version": "1.0.0",
            "hooks": [{"event": "PreToolUse", "matcher": "Bash(rm *)", "command": "guard.sh"}],
            "skills_paths": ["skills"],
        },
        skills={"commit.md": "---\ndescription: commit\n---\nCommit: $ARGUMENTS\n"},
    )

    plugins = await load_plugins(tmp_path)

    assert len(plugins) == 1
    assert plugins[0].manifest.name == "git-tools"
    # Hook merged into the global registry.
    assert any(h.command == "guard.sh" for h in get_hooks())
    # Skill registered.
    assert get_skill("commit") is not None


async def test_disabled_plugin_is_skipped(tmp_path):
    _make_plugin(
        tmp_path,
        "off",
        {
            "name": "off",
            "enabled": False,
            "hooks": [{"event": "Stop", "command": "noop.sh"}],
        },
    )
    plugins = await load_plugins(tmp_path)
    assert plugins == []
    assert get_hooks() == []


async def test_malformed_manifest_is_skipped(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text("{ not valid json")
    good = _make_plugin(tmp_path, "good", {"name": "good"})
    assert good.exists()

    plugins = await load_plugins(tmp_path)
    assert {p.manifest.name for p in plugins} == {"good"}


async def test_missing_plugins_dir_returns_empty(tmp_path):
    assert await load_plugins(tmp_path / "nope") == []


# --- load_extensions orchestrator -------------------------------------------


async def test_load_extensions_wires_settings_and_plugins(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "hello.md").write_text("---\ndescription: hi\n---\nSay hi\n")

    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    _make_plugin(
        plugins_dir,
        "p1",
        {"name": "p1", "skills_paths": ["skills"]},
        skills={"deploy.md": "---\ndescription: d\n---\nDeploy\n"},
    )

    from nano_claude.extensibility.hooks.types import HookDefinition

    settings = Settings(hooks=[HookDefinition(event="Stop", command="s.sh")])
    summary = await load_extensions(settings, skills_dir=skills_dir, plugins_dir=plugins_dir)

    assert summary.hooks == 1
    assert summary.skills == 2  # one direct + one from the plugin
    assert summary.plugins == 1
    assert get_skill("hello") is not None
    assert get_skill("deploy") is not None


# --- plugin MCP server (real stdio) -----------------------------------------

_SERVER = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("greet")

    @mcp.tool()
    def hi(name: str) -> str:
        '''Greet someone.'''
        return f"hello {name}"

    if __name__ == "__main__":
        mcp.run()
    """
)


async def test_plugin_connects_mcp_server(tmp_path):
    server = tmp_path / "greet_server.py"
    server.write_text(_SERVER)
    _make_plugin(
        tmp_path,
        "greeter",
        {
            "name": "greeter",
            "mcp_servers": {"greet": {"command": sys.executable, "args": [str(server)]}},
        },
    )

    try:
        plugins = await load_plugins(tmp_path)
        assert len(plugins) == 1
        assert any(t.name == "mcp__greet__hi" for t in plugins[0].mcp_tools)
        # Registered into the global tool registry.
        assert get_tool("mcp__greet__hi") is not None
    finally:
        await mcp_client.close_mcp()
