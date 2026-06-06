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
    # verification extension: lint/typecheck + discover commands
    assert "lint" in prompt
    assert "discover it from the repo" in prompt
    # diagnose-before-retrying (anti-thrash)
    assert "diagnose why before trying again" in prompt
    assert "step back and reconsider the root cause" in prompt


def test_system_prompt_includes_conventions(tmp_path):
    prompt = build_system_prompt(str(tmp_path)).lower()
    assert "follow the conventions" in prompt
    assert "already a dependency" in prompt
    assert "nothing more, nothing less" in prompt
    # simplicity-first / anti-overengineering
    assert "no speculative features" in prompt
    # surgical changes: no drive-by refactor, clean only your own orphans
    assert "keep changes surgical" in prompt
    assert "clean up only your own mess" in prompt


def test_system_prompt_includes_think_before_coding(tmp_path):
    prompt = build_system_prompt(str(tmp_path)).lower()
    # ask when interactive, else state the assumption (eval-safe)
    assert "don't guess silently" in prompt
    assert "state the assumption" in prompt
    # don't edit code you don't understand
    assert "don't edit code you don't understand" in prompt


def test_system_prompt_includes_reproduce_then_fix(tmp_path):
    prompt = build_system_prompt(str(tmp_path)).lower()
    assert "write or identify a test that reproduces it" in prompt


def test_system_prompt_includes_tool_batching(tmp_path):
    prompt = build_system_prompt(str(tmp_path)).lower()
    assert "multiple tool calls in a single turn" in prompt
    assert "run in parallel" in prompt
