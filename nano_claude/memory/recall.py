"""On-demand memory recall: pick the few relevant files and surface them.

``MEMORY.md`` is always in the system prompt; full topic files are surfaced
only when relevant, chosen by a cheap side-query. Recall runs as a non-blocking
``asyncio.Task`` fired once per user turn (:class:`MemorySession.start`) and is
consumed by the loop only once it has *settled* — it never stalls the turn.
Mirrors ``findRelevantMemories.ts`` + the ``attachments.ts`` surfacing path.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import litellm

from nano_claude.memory.age import memory_age, memory_freshness_text
from nano_claude.memory.scan import format_manifest, scan_memory_files

# Per-file surfacing caps and the per-session ceiling on total injected memory.
MAX_MEMORY_LINES = 200
MAX_MEMORY_BYTES = 4096
MAX_SESSION_BYTES = 60 * 1024
MAX_SELECTED = 5

SELECT_PROMPT = (
    "You select memory files that will help Claude Code answer the user's query. "
    "You get the query and a list of available memories (filename + description). "
    'Return JSON {"selected": [filenames]} with up to 5 files you are confident '
    "are useful — be selective; return an empty list if none clearly help. "
    "If recently-used tools are listed, do NOT select usage/API-reference memories "
    "for those tools (they are already in use), but DO select memories describing "
    "warnings, gotchas, or known issues about them."
)


async def find_relevant_memories(
    query: str,
    mdir: Path,
    *,
    recall_model: str,
    recent_tools: Iterable[str] = (),
    already_surfaced: frozenset[str] = frozenset(),
) -> list[Path]:
    """Return up to 5 memory-file paths relevant to ``query`` (excludes surfaced)."""
    headers = [h for h in scan_memory_files(mdir) if str(h.path) not in already_surfaced]
    if not headers:
        return []

    tools = list(recent_tools)
    tools_line = f"\n\nRecently used tools: {', '.join(tools)}" if tools else ""
    user = f"Query: {query}\n\nAvailable memories:\n{format_manifest(headers)}{tools_line}"

    try:
        resp = await litellm.acompletion(
            model=recall_model,
            messages=[
                {"role": "system", "content": SELECT_PROMPT},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=256,
        )
        content = resp.choices[0].message.content or "{}"
        selected = set(json.loads(content).get("selected", []))
    except Exception:  # noqa: BLE001 - recall is best-effort; never surface an error to the turn
        return []

    return [h.path for h in headers if h.filename in selected][:MAX_SELECTED]


def _read_capped(path: Path) -> tuple[str, bool]:
    """Read a memory file capped to the line and byte limits; flag truncation."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", False
    truncated = False
    lines = text.splitlines()
    if len(lines) > MAX_MEMORY_LINES:
        text = "\n".join(lines[:MAX_MEMORY_LINES])
        truncated = True
    if len(text.encode("utf-8")) > MAX_MEMORY_BYTES:
        text = text.encode("utf-8")[:MAX_MEMORY_BYTES].decode("utf-8", "ignore")
        truncated = True
    return text, truncated


def surface_memory_attachment(path: Path) -> dict | None:
    """Build a ``<system-reminder>`` message carrying one memory file's content.

    Header notes how old the memory is; files older than a day add a staleness
    caveat; over-cap files note the truncation and point at the full path.
    """
    content, truncated = _read_capped(path)
    if not content.strip():
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0

    header = f"Memory (saved {memory_age(mtime)}): {path}"
    body = [header, "", content]
    freshness = memory_freshness_text(mtime)
    if freshness:
        body.append(f"\n{freshness}")
    if truncated:
        body.append(f"\n> Truncated — use the Read tool for the full file at {path}")
    reminder = "<system-reminder>\n" + "\n".join(body) + "\n</system-reminder>"
    return {"role": "user", "content": reminder}


@dataclass
class MemorySession:
    """Per-conversation recall state: which files were surfaced, and how much.

    Lives for the life of one conversation (reset on ``/clear``). ``surfaced``
    de-dups across turns; ``bytes_used`` enforces the session ceiling.
    """

    mdir: Path
    recall_model: str
    surfaced: set[str] = field(default_factory=set)
    bytes_used: int = 0

    def start(self, query: str, recent_tools: Iterable[str] = ()) -> MemoryPrefetch:
        """Fire the relevance side-query as a background task (does not await)."""
        task = asyncio.create_task(
            find_relevant_memories(
                query,
                self.mdir,
                recall_model=self.recall_model,
                recent_tools=list(recent_tools),
                already_surfaced=frozenset(self.surfaced),
            )
        )
        return MemoryPrefetch(self, task)


@dataclass
class MemoryPrefetch:
    """A fired recall, consumed only once it has settled (zero-wait)."""

    session: MemorySession
    task: asyncio.Task
    consumed: bool = False

    def drain_if_ready(self) -> list[dict]:
        """Return surfaced messages if the task is done; ``[]`` otherwise.

        Idempotent after the first successful drain. Enforces de-dup and the
        session byte cap, recording what it surfaces on the session.
        """
        if self.consumed or not self.task.done():
            return []
        self.consumed = True
        try:
            paths = self.task.result()
        except Exception:  # noqa: BLE001 - cancelled or failed recall yields nothing
            return []

        messages: list[dict] = []
        for path in paths:
            key = str(path)
            if key in self.session.surfaced:
                continue
            if self.session.bytes_used >= MAX_SESSION_BYTES:
                break
            msg = surface_memory_attachment(path)
            if msg is None:
                continue
            self.session.surfaced.add(key)
            self.session.bytes_used += len(msg["content"])
            messages.append(msg)
        return messages

    def cancel(self) -> None:
        """Cancel the underlying task if it never settled (turn ended first)."""
        if not self.task.done():
            self.task.cancel()
