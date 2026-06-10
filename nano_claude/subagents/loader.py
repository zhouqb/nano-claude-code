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

# An independent, adversarial verifier. The point of running it in a *separate*
# context is structural: a fresh agent told to "try to break it" catches the
# regressions and missed cases the implementer — invested in its own fix —
# talks itself past. It is deliberately READ-ONLY (no Edit/Write): it does not
# fix anything, it returns a verdict the caller acts on. Spawn it after you
# believe a change is complete, handing it the original task and a summary of
# what you changed.
VERIFICATION = AgentDefinition(
    name="verification",
    description=(
        "Independent, adversarial verifier for a change you believe is complete. "
        "It reproduces the reported problem, checks the fix actually resolves it, "
        "hunts for regressions and missed cases, and returns a VERDICT "
        "(PASS / PARTIAL / FAIL) with evidence. Read-only — it reports, it does "
        "not edit. Delegate to it before declaring a task done; pass the original "
        "task plus a summary of what you changed and which files."
    ),
    tools=["Read", "Grep", "Glob", "Bash"],
    system_prompt=(
        "You are an independent verification subagent. A change has been made to "
        "resolve a task and your job is to find out whether it is actually "
        "correct and complete — assume it is NOT until the evidence shows "
        "otherwise. You did not write the change, so you owe it no benefit of the "
        "doubt. You are read-only: you VERIFY, you do not fix. Do not edit files.\n\n"
        "Work from evidence, not from reading the diff and nodding:\n"
        "1. REPRODUCE: from the task description, construct the case the change "
        "claims to fix and run it. Confirm the new behavior is actually correct, "
        "not just different or non-crashing.\n"
        "2. HUNT REGRESSIONS: run the existing tests that exercise the code that "
        "was touched (search the test tree for the changed symbols). Anything "
        "that passed before and fails now is a regression — report it.\n"
        "3. ATTACK THE EDGES: try to break the fix. Other operators/types, "
        "empty / None / zero / negative / infinity, symmetric and inverse paths, "
        "the cases the task implies beyond the one it spells out. A fix that "
        "handles only the literal example is PARTIAL, not PASS.\n\n"
        "You cannot ask follow-up questions. End with a single line\n"
        "  VERDICT: PASS | PARTIAL | FAIL\n"
        "followed by the specific evidence — the commands you ran and their "
        "output, the exact cases that still fail, and what remains to be done. "
        "Be concrete; the caller acts only on what you report."
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
    """Register the built-in agents plus every agent file in ``directory``."""
    register_agent(GENERAL_PURPOSE)
    register_agent(VERIFICATION)

    directory = directory or DEFAULT_AGENTS_DIR
    loaded: list[AgentDefinition] = [GENERAL_PURPOSE, VERIFICATION]
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
