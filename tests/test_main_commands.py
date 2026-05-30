"""Tests for built-in REPL slash command prompts."""

from __future__ import annotations

from nano_claude.main import INIT_PROMPT


def test_init_prompt_targets_claude_md():
    assert "create a CLAUDE.md file" in INIT_PROMPT
    assert "If there's already a CLAUDE.md, suggest improvements" in INIT_PROMPT
    assert "# CLAUDE.md" in INIT_PROMPT


def test_init_prompt_discourages_generic_content():
    assert "Avoid listing every component or file structure" in INIT_PROMPT
    assert "Don't include generic development practices" in INIT_PROMPT
