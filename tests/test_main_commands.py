"""Tests for built-in REPL slash command prompts."""

from __future__ import annotations

from nano_claude.agent.types import AgentConfig, LoopState, TokenUsage
from nano_claude.main import INIT_PROMPT, _reset_state_for_clear


def test_init_prompt_targets_claude_md():
    assert "create a CLAUDE.md file" in INIT_PROMPT
    assert "If there's already a CLAUDE.md, suggest improvements" in INIT_PROMPT
    assert "# CLAUDE.md" in INIT_PROMPT


def test_init_prompt_discourages_generic_content():
    assert "Avoid listing every component or file structure" in INIT_PROMPT
    assert "Don't include generic development practices" in INIT_PROMPT


def test_reset_state_for_clear_starts_fresh_session(tmp_path):
    old_storage = object()
    state = LoopState(
        messages=[{"role": "user", "content": "old"}],
        turn_count=3,
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        last_input_tokens=10,
        consecutive_compact_failures=1,
        storage=old_storage,
        budget=object(),
        collapse=object(),
        last_assistant_at=123.0,
        read_file_state={"x": object()},
    )

    storage = _reset_state_for_clear(state, AgentConfig(cwd=str(tmp_path), model="test-model"))

    assert storage is state.storage
    assert storage is not old_storage
    assert storage.session_id
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == "system"
    assert "Working directory:" in state.messages[0]["content"]
    assert state.turn_count == 0
    assert state.token_usage.total == 0
    assert state.last_input_tokens == 0
    assert state.consecutive_compact_failures == 0
    assert state.budget is None
    assert state.collapse is None
    assert state.last_assistant_at is None
    assert state.read_file_state == {}
