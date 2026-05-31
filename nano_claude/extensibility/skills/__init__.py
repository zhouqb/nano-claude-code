"""Skills: ``/command`` shortcuts that expand into prompts for the model."""

from nano_claude.extensibility.skills.loader import (
    SKILL_REGISTRY,
    SkillDispatch,
    clear_skills,
    dispatch_skill,
    get_skill,
    load_skills,
    register_skill,
)
from nano_claude.extensibility.skills.types import SkillContext, SkillDefinition

__all__ = [
    "SKILL_REGISTRY",
    "SkillContext",
    "SkillDefinition",
    "SkillDispatch",
    "clear_skills",
    "dispatch_skill",
    "get_skill",
    "load_skills",
    "register_skill",
]
