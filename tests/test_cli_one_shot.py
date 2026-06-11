"""Tests for one-shot command-line execution."""

from __future__ import annotations

import json
from types import SimpleNamespace

import litellm
from click.testing import CliRunner

from nano_claude.main import cli
from nano_claude.permissions.settings import Settings
from tests.conftest import make_sequential_acompletion, text_chunk, tool_call_chunk, usage_chunk


def test_cli_prompt_runs_single_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_sequential_acompletion([[text_chunk("done"), usage_chunk(3, 2)]]),
    )
    monkeypatch.setattr(
        "nano_claude.main.Settings.load", lambda: Settings(path=tmp_path / "s.json")
    )
    monkeypatch.setattr("nano_claude.main.register_known_models", lambda: None)
    monkeypatch.setattr("nano_claude.main.init_telemetry", lambda: False)
    monkeypatch.setattr("nano_claude.main.shutdown_telemetry", lambda: None)
    monkeypatch.setattr("nano_claude.main.load_agents", lambda: None)

    async def _load_extensions(_settings):
        return SimpleNamespace(anything=False, hooks=0, skills=0, mcp_tools=0, plugins=0)

    async def _close_mcp():
        return None

    monkeypatch.setattr("nano_claude.main.load_extensions", _load_extensions)
    monkeypatch.setattr("nano_claude.main.close_mcp", _close_mcp)

    runner = CliRunner()
    result = runner.invoke(cli, ["--model", "test-model", "say", "hi"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "assistant" in result.output
    assert "done" in result.output
    assert "Type your message" not in result.output


def test_cli_prompt_defaults_to_bypass_for_one_shot(tmp_path, monkeypatch):
    target = tmp_path / "out.txt"
    write_args = json.dumps({"file_path": str(target), "content": "hello\n"})
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_sequential_acompletion(
            [
                [tool_call_chunk(0, "c1", "Write", write_args), usage_chunk(5, 3)],
                [text_chunk("written"), usage_chunk(2, 1)],
            ]
        ),
    )
    monkeypatch.setattr(
        "nano_claude.main.Settings.load", lambda: Settings(path=tmp_path / "s.json")
    )
    monkeypatch.setattr("nano_claude.main.register_known_models", lambda: None)
    monkeypatch.setattr("nano_claude.main.init_telemetry", lambda: False)
    monkeypatch.setattr("nano_claude.main.shutdown_telemetry", lambda: None)
    monkeypatch.setattr("nano_claude.main.load_agents", lambda: None)

    async def _load_extensions(_settings):
        return SimpleNamespace(anything=False, hooks=0, skills=0, mcp_tools=0, plugins=0)

    async def _close_mcp():
        return None

    async def _should_not_prompt(*args, **kwargs):
        raise AssertionError("one-shot default permission mode should bypass prompts")

    monkeypatch.setattr("nano_claude.main.load_extensions", _load_extensions)
    monkeypatch.setattr("nano_claude.main.close_mcp", _close_mcp)
    monkeypatch.setattr("nano_claude.main.make_cli_prompter", lambda **kwargs: _should_not_prompt)

    runner = CliRunner()
    result = runner.invoke(cli, ["write", "the", "file"], catch_exceptions=False)

    assert result.exit_code == 0
    assert target.read_text() == "hello\n"
    assert "Permission required" not in result.output
