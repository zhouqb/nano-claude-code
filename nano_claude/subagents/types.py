"""Agent definitions.

A subagent is described by an :class:`AgentDefinition`: a name, a description
the *model* reads to decide when to delegate, a system prompt (the markdown
body of its ``.md`` file), an optional restricted tool set, and an optional
model override.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentDefinition:
    name: str
    description: str
    system_prompt: str
    # None ⇒ all base tools except Task (recursion is capped at depth 1).
    tools: list[str] | None = None
    # None ⇒ inherit the parent's model.
    model: str | None = None
