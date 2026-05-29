"""Tests for rule matching, settings, and the permission manager."""

from __future__ import annotations

import asyncio

from nano_claude.permissions.manager import (
    PromptOutcome,
    apply_mode_transform,
    has_permission_to_use_tool,
)
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.rules import PermissionRule, first_match, matches_rule
from nano_claude.permissions.settings import Settings
from nano_claude.tools.base import PermissionDecision, ToolContext
from nano_claude.tools.bash import BashInput, BashTool
from nano_claude.tools.write import WriteInput, WriteTool


def ctx(cwd: str, mode: PermissionMode = PermissionMode.DEFAULT) -> ToolContext:
    return ToolContext(cwd=str(cwd), cancel_event=asyncio.Event(), permission_mode=mode)


# --- rules ------------------------------------------------------------------


def test_bare_tool_name_match():
    rule = PermissionRule("Bash", "allow")
    assert matches_rule(rule, "Bash", {"command": "ls"})
    assert not matches_rule(rule, "Read", {})


def test_parenthesised_arg_match():
    rule = PermissionRule("Bash(git *)", "allow")
    assert matches_rule(rule, "Bash", {"command": "git status"})
    assert not matches_rule(rule, "Bash", {"command": "rm -rf x"})


def test_first_match_returns_first():
    rules = [PermissionRule("Read", "allow"), PermissionRule("*", "deny")]
    assert first_match(rules, "Read", {}).decision == "allow"
    assert first_match(rules, "Bash", {"command": "x"}).decision == "deny"


# --- settings ---------------------------------------------------------------


def test_settings_defaults_when_missing(tmp_path):
    s = Settings.load(tmp_path / "settings.json")
    patterns = {r.pattern for r in s.allow_rules}
    assert {"Read", "GlobTool", "Grep"} <= patterns


def test_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings.load(path)
    s.add_allow_rule("Bash")
    s.save()
    reloaded = Settings.load(path)
    assert any(r.pattern == "Bash" for r in reloaded.allow_rules)


# --- mode transform ---------------------------------------------------------


def test_bypass_upgrades_ask_to_allow():
    d = apply_mode_transform(PermissionDecision("ask"), PermissionMode.BYPASS, "Bash")
    assert d.behavior == "allow"


def test_bypass_never_overrides_deny():
    d = apply_mode_transform(PermissionDecision("deny"), PermissionMode.BYPASS, "Bash")
    assert d.behavior == "deny"


def test_accept_edits_allows_write_but_not_bash():
    write = apply_mode_transform(PermissionDecision("ask"), PermissionMode.ACCEPT_EDITS, "Write")
    bash = apply_mode_transform(PermissionDecision("ask"), PermissionMode.ACCEPT_EDITS, "Bash")
    assert write.behavior == "allow"
    assert bash.behavior == "ask"


# --- manager ----------------------------------------------------------------


async def _never_prompt(tool, args, text):
    raise AssertionError("prompter should not be called")


async def test_deny_rule_wins(tmp_path):
    settings = Settings(deny_rules=[PermissionRule("Bash", "deny")], path=tmp_path / "s.json")
    decision = await has_permission_to_use_tool(
        BashTool(), BashInput(command="ls"), ctx(tmp_path), settings, _never_prompt
    )
    assert decision.behavior == "deny"


async def test_allow_rule_skips_prompt(tmp_path):
    settings = Settings(allow_rules=[PermissionRule("Write", "allow")], path=tmp_path / "s.json")
    decision = await has_permission_to_use_tool(
        WriteTool(),
        WriteInput(file_path=str(tmp_path / "x.txt"), content="x"),
        ctx(tmp_path),
        settings,
        _never_prompt,
    )
    assert decision.behavior == "allow"


async def test_ask_then_prompt_allow_once(tmp_path):
    settings = Settings(path=tmp_path / "s.json")

    async def prompter(tool, args, text):
        return PromptOutcome.ALLOW_ONCE

    decision = await has_permission_to_use_tool(
        WriteTool(),
        WriteInput(file_path=str(tmp_path / "x.txt"), content="x"),
        ctx(tmp_path),
        settings,
        prompter,
    )
    assert decision.behavior == "allow"
    # allow-once must NOT persist a rule
    assert not any(r.pattern == "Write" for r in settings.allow_rules)


async def test_ask_then_prompt_allow_always_persists(tmp_path):
    path = tmp_path / "s.json"
    settings = Settings(path=path)

    async def prompter(tool, args, text):
        return PromptOutcome.ALLOW_ALWAYS

    await has_permission_to_use_tool(
        WriteTool(),
        WriteInput(file_path=str(tmp_path / "x.txt"), content="x"),
        ctx(tmp_path),
        settings,
        prompter,
    )
    assert any(r.pattern == "Write" for r in Settings.load(path).allow_rules)


async def test_dangerous_bash_denied_without_prompt(tmp_path):
    settings = Settings(path=tmp_path / "s.json")
    decision = await has_permission_to_use_tool(
        BashTool(), BashInput(command="rm -rf /"), ctx(tmp_path), settings, _never_prompt
    )
    assert decision.behavior == "deny"


async def test_bypass_mode_allows_bash_without_prompt(tmp_path):
    settings = Settings(allow_rules=[], path=tmp_path / "s.json")
    decision = await has_permission_to_use_tool(
        BashTool(),
        BashInput(command="ls"),
        ctx(tmp_path, PermissionMode.BYPASS),
        settings,
        _never_prompt,
    )
    assert decision.behavior == "allow"
