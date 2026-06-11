"""Turn-end memory extraction: a forked agent that saves what the main one missed.

Optional and off by default. After a completed turn the manager forks a
subagent — under a custom permission gate that allows reading anywhere but
confines writes to the memory directory (never ``BYPASS``) — to record durable
memories. It runs in the background (never blocks the user), skips turns where
the main agent already saved, coalesces overlapping runs, tracks a cursor so
each turn is processed once, and is drained on shutdown. Mirrors
``src/services/extractMemories/extractMemories.ts``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from nano_claude.agent.types import AgentConfig, LoopState
from nano_claude.memory.scan import format_manifest, scan_memory_files
from nano_claude.permissions.settings import Settings
from nano_claude.tools.base import PermissionDecision, Tool, ToolContext
from nano_claude.tools.bash import is_read_only

EXTRACT_MAX_TURNS = 5
READ_ONLY_TOOLS = frozenset({"Read", "Grep", "GlobTool"})
WRITE_TOOLS = frozenset({"Write", "Edit"})
# Bash is permitted, but only for read-only commands (see the gate below).
EXTRACT_TOOLS = sorted(READ_ONLY_TOOLS | WRITE_TOOLS | {"Bash"})

EXTRACT_SYSTEM = (
    "You are a memory-extraction agent. Review the conversation excerpt and the "
    "current memory manifest, then save any DURABLE facts worth keeping for future "
    "sessions that are not already saved — following the four-type taxonomy "
    "(user / feedback / project / reference). Write each as its own topic file with "
    "frontmatter and add a one-line pointer to MEMORY.md. Do NOT save anything "
    "derivable from code or git, ephemeral task state, or duplicates of existing "
    "memories. If nothing is worth saving, do nothing and stop."
)


def _file_path_arg(args: object) -> str | None:
    if isinstance(args, dict):
        value = args.get("file_path")
    else:
        value = getattr(args, "file_path", None)
    return value if isinstance(value, str) else None


def _within(mdir: Path, file_path: str) -> bool:
    """True if ``file_path`` is the memory dir or lives inside it."""
    try:
        target = Path(file_path).resolve()
        base = mdir.resolve()
    except OSError:
        return False
    return target == base or base in target.parents


def create_memory_can_use_tool(mdir: Path):
    """A permission gate: read freely, write only inside ``mdir``, deny the rest."""

    async def gate(tool: Tool, args: object, context: ToolContext) -> PermissionDecision:
        if tool.name in READ_ONLY_TOOLS:
            return PermissionDecision(behavior="allow")
        if tool.name in WRITE_TOOLS:
            path = _file_path_arg(args)
            if path and _within(mdir, path):
                return PermissionDecision(behavior="allow")
            return PermissionDecision(
                behavior="deny", reason="extraction may only write inside the memory directory"
            )
        if tool.name == "Bash":
            command = (
                args.get("command") if isinstance(args, dict) else getattr(args, "command", "")
            )
            if command and is_read_only(command):
                return PermissionDecision(behavior="allow")
            return PermissionDecision(
                behavior="deny", reason="extraction may only run read-only Bash commands"
            )
        return PermissionDecision(
            behavior="deny", reason=f"{tool.name} is not permitted during memory extraction"
        )

    return gate


def has_memory_writes_since(messages: list[dict], since_index: int, mdir: Path) -> bool:
    """Did the main agent already write a memory file after ``since_index``?

    If so the forked extraction is redundant — the caller advances the cursor and
    skips it, keeping main-agent and background saves mutually exclusive per turn.
    """
    for message in messages[since_index:]:
        if message.get("role") != "assistant":
            continue
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") not in WRITE_TOOLS:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            path = args.get("file_path")
            if isinstance(path, str) and _within(mdir, path):
                return True
    return False


def _format_excerpt(messages: list[dict], *, max_chars: int = 8000) -> str:
    """Render a transcript slice compactly for the extraction prompt."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            names = ", ".join(tc.get("function", {}).get("name", "") for tc in m["tool_calls"])
            content = f"{content} [called: {names}]".strip()
        if content:
            parts.append(f"{role}: {content}")
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


class ExtractionManager:
    """Schedules and coalesces background extraction runs for one conversation."""

    def __init__(self, state: LoopState, config: AgentConfig, settings: Settings, mdir: Path):
        self.state = state
        self.config = config
        self.settings = settings
        self.mdir = mdir
        self.cursor = 0
        self._running = False
        self._pending = False
        self._task: asyncio.Task | None = None

    def schedule(self) -> None:
        """Request an extraction over messages since the cursor (non-blocking)."""
        if self._running:
            self._pending = True  # coalesce: run once more after the current one
            return
        self._launch()

    def _launch(self) -> None:
        messages = self.state.messages
        end = len(messages)
        if self.cursor >= end:
            return
        if has_memory_writes_since(messages, self.cursor, self.mdir):
            self.cursor = end  # main agent already saved this range — skip the fork
            return
        self._running = True
        self._task = asyncio.create_task(self._run_and_chain(end))

    async def _run_and_chain(self, end: int) -> None:
        try:
            await self._run(end)
            self.cursor = end  # advance only on success, so a failed run retries
        except Exception:  # noqa: BLE001 - extraction must never crash the session
            pass
        finally:
            self._running = False
        if self._pending:
            self._pending = False
            self._launch()

    async def _run(self, end: int) -> None:
        from nano_claude.adk.driver import run_turn as query_loop

        excerpt = _format_excerpt(self.state.messages[self.cursor : end])
        if not excerpt.strip():
            return
        manifest = format_manifest(scan_memory_files(self.mdir)) or "(none yet)"
        sub_state = LoopState(
            messages=[
                {"role": "system", "content": f"{EXTRACT_SYSTEM}\n\nMemory directory: {self.mdir}"},
                {
                    "role": "user",
                    "content": f"Existing memories:\n{manifest}\n\nConversation excerpt:\n{excerpt}",
                },
            ]
        )
        sub_config = AgentConfig(
            model=self.config.model,
            permission_mode=self.config.permission_mode,
            max_turns=EXTRACT_MAX_TURNS,
            cwd=self.config.cwd,
        )
        await query_loop(
            sub_state,
            sub_config,
            settings=self.settings,
            allowed_tools=EXTRACT_TOOLS,
            permission_override=create_memory_can_use_tool(self.mdir),
        )

    async def drain(self, timeout: float = 60.0) -> None:
        """Await any in-flight extraction on shutdown (bounded)."""
        task = self._task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except (TimeoutError, Exception):  # noqa: BLE001 - best-effort drain
            pass
