"""Tests for system prompt assembly."""

from __future__ import annotations

from nano_claude.context import build_environment_block, build_system_prompt


def test_environment_block_has_core_fields(tmp_path):
    block = build_environment_block(str(tmp_path))
    assert "Working directory:" in block
    assert "Platform:" in block
    assert "Shell:" in block
    assert "Today's date:" in block
    assert "Is a git repository:" in block


def test_system_prompt_includes_identity_and_env(tmp_path):
    prompt = build_system_prompt(str(tmp_path))
    assert "nano-claude-code" in prompt
    assert "Working directory:" in prompt


def test_system_prompt_picks_up_claude_md(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("Always write tests first.", encoding="utf-8")
    prompt = build_system_prompt(str(tmp_path))
    assert "Always write tests first." in prompt


def test_system_prompt_includes_engineering_principles(tmp_path):
    prompt = build_system_prompt(str(tmp_path)).lower()
    # take test failures seriously / don't bypass them
    assert "test failures seriously" in prompt
    assert "just to make" in prompt
    # run tests around changed code; add tests for new code
    assert "run the tests that cover" in prompt
    assert "write unit tests" in prompt
