"""Tests for the built-in REPL commands (pure render functions)."""

from __future__ import annotations

from nano_claude.agent.types import TokenUsage
from nano_claude.commands import (
    BUILTIN_COMMANDS,
    estimate_cost,
    format_cost,
    format_help,
    format_model,
    format_turn_footer,
)
from nano_claude.extensibility.skills.types import SkillDefinition
from nano_claude.subagents.types import AgentDefinition

# --- /help ------------------------------------------------------------------


def test_help_lists_all_builtins():
    text = format_help()
    for name, _ in BUILTIN_COMMANDS:
        assert name in text


def test_help_includes_skills_and_agents():
    skills = {
        "commit": SkillDefinition(
            name="commit", description="Stage and commit", argument_hint="[scope]"
        )
    }
    agents = {"explorer": AgentDefinition("explorer", "Read-only search", "prompt")}
    text = format_help(skills, agents)
    assert "/commit" in text
    assert "Stage and commit" in text
    assert "[scope]" in text
    assert "explorer" in text
    assert "Read-only search" in text


def test_help_handles_no_skills_or_agents():
    text = format_help()
    assert "Skills" not in text
    assert "Agents" not in text


# --- /cost ------------------------------------------------------------------


def test_cost_reports_token_breakdown():
    usage = TokenUsage(input_tokens=1000, output_tokens=200, cache_read_tokens=50)
    text = format_cost(usage, "anthropic/claude-sonnet-4-6")
    assert "1,000" in text
    assert "200" in text
    assert "50" in text
    assert "1,250" in text  # total


def test_cost_estimate_is_positive_for_known_model():
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    cost = estimate_cost(usage, "anthropic/claude-sonnet-4-6")
    assert cost is not None
    assert cost > 0


def test_cost_estimate_none_for_unknown_model():
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    assert estimate_cost(usage, "totally/made-up-model-xyz") is None


def test_cost_unavailable_message_for_unknown_model():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    text = format_cost(usage, "totally/made-up-model-xyz")
    assert "unavailable" in text


# --- /model -----------------------------------------------------------------


def test_model_shows_name_and_window():
    text = format_model("anthropic/claude-sonnet-4-6", 200_000)
    assert "anthropic/claude-sonnet-4-6" in text
    assert "200,000" in text


# --- turn footer ------------------------------------------------------------


def test_turn_footer_shows_total_and_cost():
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    text = format_turn_footer(usage, "anthropic/claude-sonnet-4-6")
    assert "1,500 tokens" in text
    assert "$" in text


def test_turn_footer_omits_cost_for_unknown_model():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    text = format_turn_footer(usage, "totally/made-up-model-xyz")
    assert "15 tokens" in text
    assert "$" not in text
