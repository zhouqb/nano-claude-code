"""Unit tests for the docker rollout backend's pure helpers.

The container-touching code needs Docker + swebench images, so it's exercised by
the smoke run, not here. These cover the logic we can check in isolation:
forwarding API keys by name only, and the agent command line.
"""

from __future__ import annotations

from evals.config import RolloutConfig
from evals.docker_rollout import NANO_BIN, agent_argv, present_key_names


def test_present_key_names_only_returns_set_vars():
    env = {"DEEPSEEK_API_KEY": "sk-secret", "PATH": "/usr/bin", "OPENAI_API_KEY": ""}
    names = present_key_names(env)
    assert names == ["DEEPSEEK_API_KEY"]  # empty OPENAI key and PATH excluded


def test_present_key_names_returns_names_not_values():
    env = {"ANTHROPIC_API_KEY": "sk-do-not-leak"}
    names = present_key_names(env)
    assert "ANTHROPIC_API_KEY" in names
    assert "sk-do-not-leak" not in names  # values must never be returned


def test_agent_argv_is_oneshot_nonprompting():
    cfg = RolloutConfig(model="deepseek/deepseek-v4-flash", max_turns=200)
    argv = agent_argv(cfg)
    assert argv[0] == NANO_BIN
    assert "--stdin" in argv
    assert argv[argv.index("--model") + 1] == "deepseek/deepseek-v4-flash"
    assert argv[argv.index("--max-turns") + 1] == "200"
    assert argv[argv.index("--permission-mode") + 1] == "bypassPermissions"


def test_agent_argv_omits_reasoning_effort_by_default():
    argv = agent_argv(RolloutConfig())
    assert "--reasoning-effort" not in argv  # provider default unless asked


def test_agent_argv_forwards_reasoning_effort():
    cfg = RolloutConfig(reasoning_effort="high")
    argv = agent_argv(cfg)
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
