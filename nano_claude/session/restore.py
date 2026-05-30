"""Load sessions back from disk and list resumable sessions.

``restore_messages`` rebuilds the agent loop's messages list from a session's
records (dedup by uuid). ``repair_messages`` then makes that list API-valid
after a crash or interruption. ``list_sessions`` powers the ``--resume`` picker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nano_claude.session.storage import (
    DEFAULT_ROOT,
    MessageRecord,
    MetadataRecord,
    SessionRecord,
    parse_record,
    project_dir,
)
from nano_claude.tools.base import FileReadSnapshot

INTERRUPTED = "[Interrupted]"
CONTINUE_PROMPT = "Continue from where you left off."


def load_records(path: Path) -> list[SessionRecord]:
    """Read and parse all records from a session file (skipping bad lines)."""
    if not path.is_file():
        return []
    records: list[SessionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = parse_record(line)
        if record is not None:
            records.append(record)
    return records


def restore_messages(records: list[SessionRecord]) -> list[dict[str, Any]]:
    """Rebuild the messages list from message records, deduped by uuid."""
    seen: set[str] = set()
    messages: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, MessageRecord):
            if record.uuid in seen:
                continue
            seen.add(record.uuid)
            messages.append(record.message)
    return messages


def last_assistant_ts(records: list[SessionRecord]) -> float | None:
    """Wall-clock ``ts`` of the most recent assistant message, or None if none.

    Seeds ``LoopState.last_assistant_at`` on resume so the Layer 3 microcompact
    time gate measures the real idle gap across the resume (the in-memory message
    dicts carry no timestamp; the records do).
    """
    for record in reversed(records):
        if isinstance(record, MessageRecord) and record.message.get("role") == "assistant":
            return record.ts
    return None


def repair_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make a messages list API-valid after a crash or mid-turn interruption.

    The critical invariant: every assistant ``tool_calls`` entry must be
    followed by a ``tool`` message with the matching ``tool_call_id``. If a
    crash happened after the model requested tools but before (all) results were
    recorded, inject synthetic ``[Interrupted]`` tool results for the missing
    ones so the conversation can resume.
    """
    resolved_ids = {
        m["tool_call_id"] for m in messages if m.get("role") == "tool" and "tool_call_id" in m
    }
    repaired: list[dict[str, Any]] = []
    for m in messages:
        repaired.append(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id") and tc["id"] not in resolved_ids:
                    repaired.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": INTERRUPTED}
                    )
                    resolved_ids.add(tc["id"])
    last = next(
        (
            m
            for m in reversed(repaired)
            if m.get("role") in ("user", "assistant") and m.get("content") != INTERRUPTED
        ),
        None,
    )
    if last and last.get("role") == "user":
        repaired.append({"role": "assistant", "content": INTERRUPTED})
        repaired.append({"role": "user", "content": CONTINUE_PROMPT})
    return repaired


def load_session(path: Path) -> list[dict[str, Any]]:
    """Convenience: load a session file into a repaired, resumable messages list."""
    return repair_messages(restore_messages(load_records(path)))


def restore_read_file_state(
    messages: list[dict[str, Any]], cwd: str
) -> dict[str, FileReadSnapshot]:
    """Rebuild file-read snapshots from prior full Read tool calls.

    Claude Code persists file-history snapshots. Nano's transcript only stores
    messages, so this restores the useful part best-effort by re-reading files
    that the transcript shows were read fully. If the file no longer exists, or
    the transcript shows a partial/truncated read, no snapshot is restored.
    """
    tool_results = {
        m.get("tool_call_id"): m.get("content", "")
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id")
    }
    restored: dict[str, FileReadSnapshot] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("name") != "Read":
                continue
            tool_call_id = tool_call.get("id")
            result_content = str(tool_results.get(tool_call_id, ""))
            if result_content == INTERRUPTED or "more lines truncated" in result_content:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if args.get("offset", 0) not in (0, None):
                continue
            file_path = args.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            path = Path(file_path)
            if not path.is_absolute():
                path = Path(cwd) / path
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            restored[str(path)] = FileReadSnapshot(
                content=content,
                timestamp=path.stat().st_mtime,
                offset=None,
                limit=None,
                is_partial_view=False,
            )
    return restored


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    model: str
    cwd: str
    mtime: float
    preview: str  # first user message, truncated


def _first_user_preview(records: list[SessionRecord], limit: int = 60) -> str:
    for record in records:
        if isinstance(record, MessageRecord) and record.message.get("role") == "user":
            content = record.message.get("content")
            if isinstance(content, str) and content.strip():
                text = content.strip().replace("\n", " ")
                return text if len(text) <= limit else text[:limit] + "…"
    return "(no prompt)"


def list_sessions(cwd: str, root: Path = DEFAULT_ROOT) -> list[SessionInfo]:
    """List resumable sessions for ``cwd``, most recently modified first."""
    pdir = project_dir(cwd, root)
    if not pdir.is_dir():
        return []
    infos: list[SessionInfo] = []
    for path in pdir.glob("*.jsonl"):
        records = load_records(path)
        if not records:
            continue
        meta = next((r for r in records if isinstance(r, MetadataRecord)), None)
        infos.append(
            SessionInfo(
                session_id=path.stem,
                path=path,
                model=meta.model if meta else "?",
                cwd=meta.cwd if meta else cwd,
                mtime=path.stat().st_mtime,
                preview=_first_user_preview(records),
            )
        )
    infos.sort(key=lambda i: i.mtime, reverse=True)
    return infos
