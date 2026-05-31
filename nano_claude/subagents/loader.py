"""Agent discovery and the global agent registry.

Agents are loaded once at startup from ``~/.nano-claude/agents/*.md`` (Markdown
with YAML frontmatter), plus a built-in ``general-purpose`` agent registered in
code so delegation works out of the box.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from nano_claude.subagents.types import AgentDefinition

DEFAULT_AGENTS_DIR = Path.home() / ".nano-claude" / "agents"

# Populated at startup by load_agents().
AGENT_REGISTRY: dict[str, AgentDefinition] = {}

# Always available, even with no agent files on disk.
GENERAL_PURPOSE = AgentDefinition(
    name="general-purpose",
    description=(
        "General-purpose agent for researching complex questions, searching code, "
        "and executing multi-step tasks. Use when a task needs several rounds of "
        "exploration you don't want filling the main context."
    ),
    system_prompt=(
        "You are a general-purpose subagent. Complete the delegated task "
        "autonomously using the tools available to you, then report back a "
        "concise, self-contained summary of what you found or did, including "
        "any file:line references the caller will need. You cannot ask the "
        "caller follow-up questions, so make reasonable assumptions and state "
        "them."
    ),
)


def register_agent(agent: AgentDefinition) -> None:
    AGENT_REGISTRY[agent.name] = agent


def clear_agents() -> None:
    AGENT_REGISTRY.clear()


def get_agent(name: str) -> AgentDefinition | None:
    return AGENT_REGISTRY.get(name)


def _parse_tools(raw: object) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _agent_from_markdown(path: Path) -> AgentDefinition:
    post = frontmatter.load(str(path))
    meta = post.metadata
    return AgentDefinition(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description", "")),
        system_prompt=post.content.strip(),
        tools=_parse_tools(meta.get("tools")),
        model=(str(meta["model"]) if meta.get("model") else None),
    )


def load_agents(directory: Path | None = None) -> list[AgentDefinition]:
    """Register the built-in agent plus every agent file in ``directory``."""
    register_agent(GENERAL_PURPOSE)

    directory = directory or DEFAULT_AGENTS_DIR
    loaded: list[AgentDefinition] = [GENERAL_PURPOSE]
    if not directory.is_dir():
        return loaded

    for path in sorted(directory.iterdir()):
        if path.suffix != ".md":
            continue
        try:
            agent = _agent_from_markdown(path)
        except Exception:  # noqa: BLE001 - one bad agent must not break startup
            continue
        register_agent(agent)
        loaded.append(agent)
    return loaded
