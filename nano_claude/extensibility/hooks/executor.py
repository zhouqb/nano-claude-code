"""Hook execution and the global hook registry.

``execute_hooks`` is the single entry point the loop calls at each lifecycle
event. It feeds each matching hook a JSON payload on stdin (mirroring Claude
Code's hook protocol) and interprets the exit code:

- ``0``      → success. stdout is collected as additional context.
- ``2``      → blocking error. For ``PreToolUse`` this denies the tool call;
               for other events the stderr is surfaced to the model as feedback.
- other ``≠0``→ non-blocking error. stderr is surfaced as a warning; execution
               continues.

The registry is a process-global list (``HOOK_REGISTRY``) populated once at
startup by the settings loader and plugin loader — the same model Claude Code
uses. Tests populate it via ``register_hooks`` / ``clear_hooks``.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field

from nano_claude.extensibility.hooks.types import HookDefinition, HookEvent
from nano_claude.permissions.rules import matches_pattern

# Populated by the settings loader and plugin loader at startup.
HOOK_REGISTRY: list[HookDefinition] = []


def register_hooks(hooks: list[HookDefinition]) -> None:
    HOOK_REGISTRY.extend(hooks)


def clear_hooks() -> None:
    HOOK_REGISTRY.clear()


def get_hooks(event: HookEvent | None = None) -> list[HookDefinition]:
    if event is None:
        return list(HOOK_REGISTRY)
    return [h for h in HOOK_REGISTRY if h.event == event]


@dataclass
class HookOutcome:
    """Aggregate result of firing all hooks for one event."""

    blocked: bool = False  # a PreToolUse hook denied the call
    block_reason: str = ""  # message to show the model when blocked
    # stdout / blocking stderr collected from hooks, joined for the model to see
    # (PostToolUse appends this to the tool result; SessionStart injects it).
    context: list[str] = field(default_factory=list)
    # Non-blocking warnings (other non-zero exits) for the UI; never block.
    warnings: list[str] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return "\n".join(self.context)


def _build_payload(
    event: HookEvent,
    *,
    session_id: str,
    cwd: str,
    tool_name: str | None,
    tool_input: dict | None,
    tool_response: str | None,
) -> dict:
    """Construct the stdin JSON payload, mirroring Claude Code's field names."""
    payload: dict = {
        "session_id": session_id,
        "cwd": cwd,
        "hook_event_name": event.value,
    }
    if tool_name is not None:
        payload["tool_name"] = tool_name
    if tool_input is not None:
        payload["tool_input"] = tool_input
    if tool_response is not None:
        payload["tool_response"] = tool_response
    return payload


async def _run_one(hook: HookDefinition, payload: dict) -> tuple[int, str, str]:
    """Run a single hook command, feeding ``payload`` on stdin.

    Returns ``(exit_code, stdout, stderr)``. A spawn failure or timeout maps to
    exit code 1 (a non-blocking error) with the reason on stderr.
    """
    proc = await asyncio.create_subprocess_shell(
        hook.command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "NANO_CLAUDE_TOOL": payload.get("tool_name", "")},
    )
    data = (json.dumps(payload) + "\n").encode()
    try:
        out, err = await asyncio.wait_for(proc.communicate(data), timeout=hook.timeout_s)
    except TimeoutError:
        proc.kill()
        return 1, "", f"hook timed out after {hook.timeout_s}s"
    except OSError as exc:
        return 1, "", f"hook failed to run: {exc}"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


async def execute_hooks(
    event: HookEvent,
    *,
    session_id: str = "",
    cwd: str = "",
    tool_name: str | None = None,
    tool_input: dict | None = None,
    tool_response: str | None = None,
) -> HookOutcome:
    """Fire every registered hook matching ``event`` (and the tool, if any)."""
    outcome = HookOutcome()
    payload = _build_payload(
        event,
        session_id=session_id,
        cwd=cwd,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_response=tool_response,
    )

    for hook in get_hooks(event):
        # Tool events honour the matcher; lifecycle events ignore it.
        if tool_name is not None and hook.matcher not in (None, "", "*"):
            if not matches_pattern(hook.matcher, tool_name, tool_input or {}):
                continue

        if hook.run_async:
            # Fire-and-forget: never blocks, output ignored.
            asyncio.create_task(_run_one(hook, payload))  # noqa: RUF006
            continue

        code, out, err = await _run_one(hook, payload)
        out, err = out.strip(), err.strip()

        if code == 2:
            # Blocking error. PreToolUse denies the call; everything else
            # surfaces the message back to the model as feedback context.
            reason = err or out or "Blocked by hook"
            if event is HookEvent.PRE_TOOL_USE:
                outcome.blocked = True
                outcome.block_reason = reason
                return outcome  # first deny wins; don't run remaining hooks
            outcome.context.append(reason)
        elif code != 0:
            if err:
                outcome.warnings.append(err)
        if out:
            outcome.context.append(out)

    return outcome
