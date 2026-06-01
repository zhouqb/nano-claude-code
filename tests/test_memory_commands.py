"""Tests for the /memory, /remember, and /forget built-in commands."""

from __future__ import annotations

from nano_claude.commands import (
    BUILTIN_COMMANDS,
    forget_directive,
    format_memory,
    remember_directive,
)
from nano_claude.memory.store import add_index_pointer, write_memory


def test_commands_listed_in_help_registry():
    names = {name for name, _ in BUILTIN_COMMANDS}
    assert {"/memory", "/remember", "/forget"} <= names


def test_format_memory_disabled():
    assert "disabled" in format_memory(None)


def test_format_memory_empty(tmp_path):
    out = format_memory(tmp_path)
    assert "no memories yet" in out
    assert str(tmp_path) in out


def test_format_memory_lists_files(tmp_path):
    write_memory(tmp_path, "uv", description="uses uv not pip", type="user", body="x")
    add_index_pointer(tmp_path, "uv", "uv.md")
    out = format_memory(tmp_path)
    assert "uv.md" in out
    assert "uses uv not pip" in out
    assert "[user]" in out


def test_remember_directive_includes_text_and_types():
    directive = remember_directive("we deploy on Fridays")
    assert "we deploy on Fridays" in directive
    assert "user" in directive and "reference" in directive


def test_forget_directive_includes_topic_and_pointer():
    directive = forget_directive("deploy schedule")
    assert "deploy schedule" in directive
    assert "MEMORY.md" in directive
