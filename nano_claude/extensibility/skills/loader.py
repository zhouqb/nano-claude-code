"""Skill discovery, the global registry, and ``/command`` dispatch.

Skills are loaded once at startup from ``~/.nano-claude/skills/`` (and from
plugins). At input time, ``dispatch_skill`` checks whether a line is a
registered ``/command`` and, if so, expands it into the prompt to send.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from nano_claude.extensibility.skills.types import SkillContext, SkillDefinition

DEFAULT_SKILLS_DIR = Path.home() / ".nano-claude" / "skills"

# Populated at startup by the skills loader and plugin loader.
SKILL_REGISTRY: dict[str, SkillDefinition] = {}


def register_skill(skill: SkillDefinition) -> None:
    SKILL_REGISTRY[skill.name] = skill


def clear_skills() -> None:
    SKILL_REGISTRY.clear()


def get_skill(name: str) -> SkillDefinition | None:
    return SKILL_REGISTRY.get(name)


def _parse_allowed_tools(raw: object) -> list[str] | None:
    """Accept either a YAML list or a comma-separated string; None ⇒ unrestricted."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _skill_from_markdown(path: Path) -> SkillDefinition:
    """Build a skill from a ``<name>.md`` file with YAML frontmatter."""
    post = frontmatter.load(str(path))
    meta = post.metadata
    name = str(meta.get("name") or path.stem)
    body = post.content

    async def get_prompt(arguments: str, _ctx: SkillContext, _body: str = body) -> str:
        return _body.replace("$ARGUMENTS", arguments)

    return SkillDefinition(
        name=name,
        description=str(meta.get("description", "")),
        argument_hint=(str(meta["argument-hint"]) if meta.get("argument-hint") else None),
        allowed_tools=_parse_allowed_tools(meta.get("allowed-tools")),
        get_prompt=get_prompt,
    )


def _skill_from_python(path: Path) -> SkillDefinition | None:
    """Load a ``<name>.py`` skill exposing a module-level ``SKILL``."""
    spec = importlib.util.spec_from_file_location(f"nano_skill_{path.stem}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    skill = getattr(module, "SKILL", None)
    return skill if isinstance(skill, SkillDefinition) else None


def load_skills(directory: Path | None = None) -> list[SkillDefinition]:
    """Discover skills in ``directory`` and register them. Returns what loaded."""
    directory = directory or DEFAULT_SKILLS_DIR
    if not directory.is_dir():
        return []

    loaded: list[SkillDefinition] = []
    for path in sorted(directory.iterdir()):
        try:
            if path.suffix == ".md":
                skill = _skill_from_markdown(path)
            elif path.suffix == ".py":
                skill = _skill_from_python(path)
            else:
                continue
        except Exception:  # noqa: BLE001 - one bad skill must not break startup
            continue
        if skill is not None:
            register_skill(skill)
            loaded.append(skill)
    return loaded


@dataclass
class SkillDispatch:
    """The expansion of a matched ``/command`` line."""

    prompt: str
    allowed_tools: list[str] | None


async def dispatch_skill(line: str, context: SkillContext) -> SkillDispatch | None:
    """Expand ``line`` if it is a registered ``/command``; else return None.

    A return of ``None`` means "not a skill" — the caller sends the line as-is.
    """
    if not line.startswith("/"):
        return None
    rest = line[1:]
    name, _, arguments = rest.partition(" ")
    skill = get_skill(name)
    if skill is None:
        return None
    prompt = await skill.build_prompt(arguments.strip(), context)
    return SkillDispatch(prompt=prompt, allowed_tools=skill.allowed_tools)
