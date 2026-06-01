"""Tests for the memory system-prompt section and its context wiring."""

from __future__ import annotations

from types import SimpleNamespace

from nano_claude.context import build_system_prompt
from nano_claude.memory.prompt import build_memory_section
from nano_claude.memory.store import add_index_pointer


def _settings(**extra) -> SimpleNamespace:
    return SimpleNamespace(extra=extra)


# --- build_memory_section ---------------------------------------------------


def test_section_has_taxonomy_and_save_flow(tmp_path):
    section = build_memory_section(tmp_path)
    for kind in ("user", "feedback", "project", "reference"):
        assert f"<name>{kind}</name>" in section
    assert "What NOT to save" in section
    assert "How to save memories" in section
    assert str(tmp_path) in section  # the model is told where its memory dir is


def test_section_reports_empty_index(tmp_path):
    assert "is empty" in build_memory_section(tmp_path)


def test_section_includes_index_content(tmp_path):
    add_index_pointer(tmp_path, "User prefers uv", "user-prefers-uv.md", "uv not pip")
    section = build_memory_section(tmp_path)
    assert "user-prefers-uv.md" in section
    assert "is empty" not in section


def test_building_section_creates_the_dir(tmp_path):
    mdir = tmp_path / "mem"
    build_memory_section(mdir)
    assert mdir.is_dir()


# --- context wiring (gated) -------------------------------------------------


def test_system_prompt_omits_memory_without_settings(tmp_path):
    # Existing behaviour: no settings => no memory section.
    prompt = build_system_prompt(str(tmp_path))
    assert "# Memory" not in prompt


def test_system_prompt_includes_memory_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("NANO_CLAUDE_MEMORY_DIR", str(tmp_path / "mem"))
    prompt = build_system_prompt(str(tmp_path), _settings())
    assert "# Memory" in prompt


def test_system_prompt_omits_memory_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("NANO_CLAUDE_MEMORY_DIR", str(tmp_path / "mem"))
    prompt = build_system_prompt(str(tmp_path), _settings(memoryEnabled=False))
    assert "# Memory" not in prompt
