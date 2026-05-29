"""Load sessions back from disk and list resumable sessions.

``restore_messages`` rebuilds the agent loop's messages list from a session's
records (dedup by uuid). ``repair_messages`` then makes that list API-valid
after a crash or interruption. ``list_sessions`` powers the ``--resume`` picker.
"""

from __future__ import annotations

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

INTERRUPTED = "[Interrupted]"


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
    return repaired


def load_session(path: Path) -> list[dict[str, Any]]:
    """Convenience: load a session file into a repaired, resumable messages list."""
    return repair_messages(restore_messages(load_records(path)))


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
