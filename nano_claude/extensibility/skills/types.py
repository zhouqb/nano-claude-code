"""Skill definitions.

A skill is a ``/command-name`` shortcut that expands into a prompt for the
model. Two equivalent forms populate the same registry:

- **Markdown** (``<name>.md`` with YAML frontmatter) — the body is a prompt
  template with ``$ARGUMENTS`` substituted from the text after ``/<name>``.
- **Python** (``<name>.py`` exposing a module-level ``SKILL``) — for prompts
  that must run code (e.g. read git state) to build their text.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class SkillContext:
    cwd: str
    session_id: str


# Given (arguments-after-the-command, context) -> the prompt text to send.
PromptBuilder = Callable[[str, SkillContext], Awaitable[str]]


@dataclass
class SkillDefinition:
    name: str
    description: str
    argument_hint: str | None = None
    # None ⇒ the model keeps its full tool set for this skill's turns.
    allowed_tools: list[str] | None = None
    get_prompt: PromptBuilder | None = None

    async def build_prompt(self, arguments: str, context: SkillContext) -> str:
        if self.get_prompt is None:
            return arguments
        return await self.get_prompt(arguments, context)
