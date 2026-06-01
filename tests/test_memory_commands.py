"""Tests for the /memory, /remember, and /forget built-in commands."""

from __future__ import annotations

from nano_claude.commands import (
    BUILTIN_COMMANDS,
    forget_directive,
    format_memory,
    memory_target_path,
    open_memory_file,
    remember_directive,
)
from nano_claude.memory.paths import ENTRYPOINT
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


# --- /memory <file> editor open ---------------------------------------------


def test_memory_target_defaults_to_index(tmp_path):
    assert memory_target_path(tmp_path, "") == tmp_path / ENTRYPOINT


def test_memory_target_adds_md_suffix(tmp_path):
    assert memory_target_path(tmp_path, "deploy") == tmp_path / "deploy.md"
    assert memory_target_path(tmp_path, "deploy.md") == tmp_path / "deploy.md"


def test_memory_target_cannot_escape_dir(tmp_path):
    # Directory components are stripped, so the target stays inside mdir.
    assert memory_target_path(tmp_path, "../../etc/passwd").parent == tmp_path


def test_open_memory_file_creates_and_invokes_editor(tmp_path):
    opened: dict = {}

    def fake_editor(*, filename):
        opened["filename"] = filename

    target = open_memory_file(tmp_path, "deploy", editor=fake_editor)
    assert target == tmp_path / "deploy.md"
    assert target.exists()  # pre-created so the editor opens a real file
    assert opened["filename"] == str(target)
