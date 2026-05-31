"""Spawn and run a single subagent loop.

A subagent runs in a wholly separate :class:`LoopState` (fresh message list),
with a restricted tool set that never includes ``Task`` (so recursion is capped
at depth 1), optionally a different model, and the *parent's* ``cancel_event``
so one Ctrl-C stops everything. Only the subagent's final assistant text
crosses back to the parent; its intermediate messages never enter the parent
transcript. Token usage rolls up to the parent for ``/cost`` accounting.

``query_loop`` and the tool registry are imported lazily inside the functions
to break the registry → Task tool → runner → loop → registry import cycle.
"""

from __future__ import annotations

from nano_claude.agent.types import AgentConfig, LoopResult, LoopState
from nano_claude.subagents.types import AgentDefinition
from nano_claude.tools.base import ToolContext

SUBAGENT_MAX_TURNS = 20


def _resolve_allowed_tools(agent: AgentDefinition, mode) -> list[str]:
    """Build the subagent's allowed-tool name list — always excluding Task."""
    from nano_claude.tools.registry import get_tools

    all_names = [t.name for t in get_tools(mode) if t.name != "Task"]
    if agent.tools is None:
        return all_names
    return [name for name in agent.tools if name in all_names]


def build_subagent_prompt(agent: AgentDefinition) -> str:
    return agent.system_prompt


async def run_subagent_loop(agent: AgentDefinition, prompt: str, parent: ToolContext) -> LoopResult:
    """Run ``agent`` on ``prompt`` in an isolated loop and return its result."""
    from nano_claude.agent.loop import query_loop

    sub_config = AgentConfig(
        model=agent.model or parent.parent_model or AgentConfig.model,
        permission_mode=parent.permission_mode,
        max_turns=SUBAGENT_MAX_TURNS,
        cwd=parent.cwd,
    )
    sub_state = LoopState(
        messages=[
            {"role": "system", "content": build_subagent_prompt(agent)},
            {"role": "user", "content": prompt},
        ],
        cancel_event=parent.cancel_event,  # shared ⇒ Ctrl-C aborts parent + children
    )

    result = await query_loop(
        sub_state,
        sub_config,
        settings=parent.settings,
        prompter=parent.prompter,
        allowed_tools=_resolve_allowed_tools(agent, sub_config.permission_mode),
    )

    # Roll the subagent's cost up to the parent's running total.
    if parent.token_usage_sink is not None:
        parent.token_usage_sink.merge(sub_state.token_usage)
    return result
