"""Central permission resolution.

``has_permission_to_use_tool`` is the single gate every tool call passes through
before execution. Order of resolution:

1. always-deny rules win outright
2. always-allow rules
3. the tool's own ``check_permissions``
4. permission-mode transform (never overrides a hard ``deny``)
5. if still ``ask``: prompt the user; persist "always" choices
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum, auto

from pydantic import BaseModel

from nano_claude.permissions.modes import SAFE_EDIT_TOOLS, PermissionMode
from nano_claude.permissions.rules import first_match
from nano_claude.permissions.settings import Settings
from nano_claude.tools.base import PermissionDecision, Tool, ToolContext


class PromptOutcome(Enum):
    ALLOW_ONCE = auto()
    ALLOW_ALWAYS = auto()
    DENY_ONCE = auto()
    DENY_ALWAYS = auto()


# Given (tool, args dict, prompt text) -> the user's choice.
Prompter = Callable[[Tool, dict, str], Awaitable[PromptOutcome]]


def apply_mode_transform(
    decision: PermissionDecision, mode: PermissionMode, tool_name: str
) -> PermissionDecision:
    """Upgrade an ``ask`` to ``allow`` per mode; never override a ``deny``."""
    if decision.behavior == "deny":
        return decision
    if mode == PermissionMode.BYPASS:
        return PermissionDecision(behavior="allow", reason="bypass mode")
    if mode == PermissionMode.ACCEPT_EDITS and tool_name in SAFE_EDIT_TOOLS:
        return PermissionDecision(behavior="allow", reason="acceptEdits mode")
    return decision


async def has_permission_to_use_tool(
    tool: Tool,
    args: BaseModel | dict,
    context: ToolContext,
    settings: Settings,
    prompter: Prompter,
) -> PermissionDecision:
    # MCP tools pass a raw dict; built-ins pass a Pydantic model.
    args_dict = args.model_dump() if isinstance(args, BaseModel) else dict(args)

    deny = first_match(settings.deny_rules, tool.name, args_dict)
    if deny is not None:
        return PermissionDecision(behavior="deny", reason=f"matched deny rule '{deny.pattern}'")

    allow = first_match(settings.allow_rules, tool.name, args_dict)
    if allow is not None:
        return PermissionDecision(behavior="allow", reason=f"matched allow rule '{allow.pattern}'")

    decision = await tool.check_permissions(args, context)
    decision = apply_mode_transform(decision, context.permission_mode, tool.name)
    if decision.behavior != "ask":
        return decision

    outcome = await prompter(tool, args_dict, decision.prompt)
    if outcome is PromptOutcome.ALLOW_ALWAYS:
        settings.add_allow_rule(tool.name)
        settings.save()
        return PermissionDecision(behavior="allow", reason="user allowed (always)")
    if outcome is PromptOutcome.ALLOW_ONCE:
        return PermissionDecision(behavior="allow", reason="user allowed (once)")
    if outcome is PromptOutcome.DENY_ALWAYS:
        settings.add_deny_rule(tool.name)
        settings.save()
        return PermissionDecision(behavior="deny", reason="user denied (always)")
    return PermissionDecision(behavior="deny", reason="user denied")
