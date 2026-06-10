"""Subagents: single-level, synchronous delegation via the Task tool."""

from nano_claude.subagents.loader import (
    AGENT_REGISTRY,
    GENERAL_PURPOSE,
    VERIFICATION,
    clear_agents,
    get_agent,
    load_agents,
    register_agent,
)
from nano_claude.subagents.runner import run_subagent_loop
from nano_claude.subagents.types import AgentDefinition

__all__ = [
    "AGENT_REGISTRY",
    "GENERAL_PURPOSE",
    "VERIFICATION",
    "AgentDefinition",
    "clear_agents",
    "get_agent",
    "load_agents",
    "register_agent",
    "run_subagent_loop",
]
