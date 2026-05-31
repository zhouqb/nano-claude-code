"""Tests for the skills subsystem (/command dispatch + loaders)."""

from __future__ import annotations

import litellm
import pytest

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.extensibility.skills import (
    SkillContext,
    SkillDefinition,
    clear_skills,
    dispatch_skill,
    get_skill,
    load_skills,
    register_skill,
)
from nano_claude.permissions.settings import Settings
from tests.conftest import FakeStream, text_chunk, usage_chunk


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_skills()
    yield
    clear_skills()


def _ctx() -> SkillContext:
    return SkillContext(cwd="/proj", session_id="sid")


# --- markdown loading -------------------------------------------------------


def _write_md(directory, name, text):
    (directory / f"{name}.md").write_text(text)


def test_loads_markdown_skill_with_frontmatter(tmp_path):
    _write_md(
        tmp_path,
        "commit",
        "---\n"
        "name: commit\n"
        "description: Stage and commit\n"
        "argument-hint: '[scope]'\n"
        "allowed-tools: [Bash, Read]\n"
        "---\n"
        "Create a git commit. Scope: $ARGUMENTS\n",
    )
    loaded = load_skills(tmp_path)
    assert len(loaded) == 1
    skill = get_skill("commit")
    assert skill.description == "Stage and commit"
    assert skill.argument_hint == "[scope]"
    assert skill.allowed_tools == ["Bash", "Read"]


def test_name_defaults_to_filename(tmp_path):
    _write_md(tmp_path, "deploy", "---\ndescription: ship it\n---\nDeploy now.\n")
    load_skills(tmp_path)
    assert get_skill("deploy") is not None


def test_allowed_tools_accepts_comma_string(tmp_path):
    _write_md(tmp_path, "x", "---\nallowed-tools: Bash, Grep\n---\nbody\n")
    load_skills(tmp_path)
    assert get_skill("x").allowed_tools == ["Bash", "Grep"]


def test_missing_dir_returns_empty(tmp_path):
    assert load_skills(tmp_path / "nope") == []


def test_bad_skill_does_not_break_loading(tmp_path):
    (tmp_path / "good.md").write_text("---\ndescription: ok\n---\nhi\n")
    (tmp_path / "bad.py").write_text("raise RuntimeError('boom')\n")
    loaded = load_skills(tmp_path)
    assert {s.name for s in loaded} == {"good"}


# --- python loading ---------------------------------------------------------


def test_loads_python_skill(tmp_path):
    (tmp_path / "now.py").write_text(
        "from nano_claude.extensibility.skills import SkillDefinition\n"
        "async def _p(args, ctx):\n"
        "    return f'do {args} in {ctx.cwd}'\n"
        "SKILL = SkillDefinition(name='now', description='d', get_prompt=_p)\n"
    )
    load_skills(tmp_path)
    assert get_skill("now") is not None


# --- dispatch ---------------------------------------------------------------


async def test_dispatch_substitutes_arguments(tmp_path):
    _write_md(tmp_path, "commit", "---\ndescription: c\n---\nCommit scope: $ARGUMENTS\n")
    load_skills(tmp_path)
    result = await dispatch_skill("/commit auth-module", _ctx())
    assert result is not None
    assert result.prompt == "Commit scope: auth-module"


async def test_dispatch_passes_allowed_tools(tmp_path):
    _write_md(tmp_path, "c", "---\ndescription: c\nallowed-tools: [Read]\n---\nbody\n")
    load_skills(tmp_path)
    result = await dispatch_skill("/c", _ctx())
    assert result.allowed_tools == ["Read"]


async def test_non_command_line_is_passthrough():
    assert await dispatch_skill("just a message", _ctx()) is None


async def test_unknown_command_is_passthrough():
    assert await dispatch_skill("/nonexistent foo", _ctx()) is None


async def test_python_skill_builds_prompt_from_code():
    async def _p(args, ctx):
        return f"ran {args}"

    register_skill(SkillDefinition(name="run", description="d", get_prompt=_p))
    result = await dispatch_skill("/run things", _ctx())
    assert result.prompt == "ran things"


# --- allowed_tools restriction in the loop ----------------------------------


async def test_allowed_tools_restricts_advertised_set(monkeypatch):
    captured: dict = {}

    async def _capturing_acompletion(*args, **kwargs):
        captured["tools"] = kwargs.get("tools")
        return FakeStream([text_chunk("done"), usage_chunk(1, 1)])

    monkeypatch.setattr(litellm, "acompletion", _capturing_acompletion)

    state = LoopState(messages=[{"role": "user", "content": "hi"}])
    result = await query_loop(state, AgentConfig(), settings=Settings(), allowed_tools=["Read"])

    assert result.reason is StopReason.COMPLETED
    advertised = {t["function"]["name"] for t in captured["tools"]}
    assert advertised == {"Read"}
