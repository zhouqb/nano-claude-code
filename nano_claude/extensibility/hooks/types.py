"""Hook definitions.

A hook is a shell command fired at a lifecycle event. The four events nano
supports mirror the load-bearing subset of Claude Code's larger set:

- ``PreToolUse``  — before a tool runs; exit code 2 blocks the call (deny).
- ``PostToolUse`` — after a tool runs; stdout is appended to the tool result.
- ``SessionStart``— once before the first turn; stdout becomes a context note.
- ``Stop``        — when the loop finishes a turn with no further tool calls.

Hooks are configured under ``"hooks"`` in ``~/.nano-claude/settings.json`` or
contributed by plugins, and matched against the current tool via the same
``"Bash(git *)"`` grammar the permission rules use.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class HookEvent(StrEnum):
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    SESSION_START = "SessionStart"
    STOP = "Stop"


class HookDefinition(BaseModel):
    # ``async`` is a Python keyword, so the field is ``run_async`` with an alias
    # so settings/plugin JSON can still spell it ``"async"``.
    model_config = ConfigDict(populate_by_name=True)

    event: HookEvent
    command: str = Field(description="Shell command to run.")
    matcher: str | None = Field(
        default=None,
        description="Tool-name pattern, e.g. 'Bash(git *)'. None/'*' matches every tool. "
        "Only meaningful for the tool events (PreToolUse/PostToolUse).",
    )
    run_async: bool = Field(
        default=False,
        alias="async",
        description="Fire-and-forget: run without blocking and ignore output.",
    )
    timeout_s: float = Field(
        default=60.0,
        alias="timeout",
        description="Per-command timeout in seconds (matches Claude Code's units).",
    )
