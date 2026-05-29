"""Append-only JSONL session storage with a debounced write queue.

Each session is one ``.jsonl`` file under
``~/.nano-claude/projects/<sanitized-cwd>/<session-id>.jsonl``; every line is one
record. Messages are stored as the raw OpenAI-format dicts the agent loop uses,
so restore is a lossless round-trip (see ``session/restore.py``).

Records carry a ``uuid`` so re-appends are idempotent (dedup happens on load),
and a ``ts`` for ordering/recency. Writes are batched: ``enqueue`` schedules a
flush ~0.1s later, and callers ``await flush()`` on graceful exit.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid as uuidlib
from pathlib import Path
from typing import Annotated, Any, Literal

import aiofiles
from pydantic import BaseModel, Field, TypeAdapter

DEFAULT_ROOT = Path.home() / ".nano-claude"


class MessageRecord(BaseModel):
    type: Literal["message"] = "message"
    uuid: str
    ts: float
    message: dict[str, Any]  # raw OpenAI-format message dict


class MetadataRecord(BaseModel):
    type: Literal["metadata"] = "metadata"
    uuid: str
    ts: float
    session_id: str
    model: str
    cwd: str


class CompactBoundaryRecord(BaseModel):
    type: Literal["compact_boundary"] = "compact_boundary"
    uuid: str
    ts: float
    summary: str
    pre_turn_count: int


SessionRecord = Annotated[
    MessageRecord | MetadataRecord | CompactBoundaryRecord,
    Field(discriminator="type"),
]

_ADAPTER: TypeAdapter[SessionRecord] = TypeAdapter(SessionRecord)


def parse_record(line: str) -> SessionRecord | None:
    """Parse one JSONL line into a record, or None if it's blank/corrupt."""
    line = line.strip()
    if not line:
        return None
    try:
        return _ADAPTER.validate_json(line)
    except ValueError:
        return None


def new_uuid() -> str:
    return uuidlib.uuid4().hex


def new_session_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuidlib.uuid4().hex[:8]}"


def sanitize_cwd(cwd: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", cwd).strip("-")
    return slug or "root"


def project_dir(cwd: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / "projects" / sanitize_cwd(cwd)


def session_file(cwd: str, session_id: str, root: Path = DEFAULT_ROOT) -> Path:
    return project_dir(cwd, root) / f"{session_id}.jsonl"


def session_output_dir(storage: SessionStorage | None) -> Path | None:
    """Where tools/compaction spill large output: a session-scoped folder.

    Lives beside the session JSONL so spills are cleaned up with it. Returns
    None when there's no session storage (callers fall back to a temp dir).
    """
    if storage is None:
        return None
    return storage.path.parent / f"{storage.session_id}-outputs"


class SessionStorage:
    """Buffers records and appends them to a session JSONL file."""

    FLUSH_DELAY_S = 0.1

    def __init__(self, path: Path, session_id: str):
        self.path = path
        self.session_id = session_id
        self._pending: list[SessionRecord] = []
        self._flush_handle: asyncio.TimerHandle | None = None

    def enqueue(self, entry: SessionRecord) -> None:
        self._pending.append(entry)
        if self._flush_handle is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # no loop yet; caller will flush() explicitly
            self._flush_handle = loop.call_later(
                self.FLUSH_DELAY_S, lambda: asyncio.create_task(self.flush())
            )

    def append_message(self, message: dict[str, Any]) -> MessageRecord:
        record = MessageRecord(uuid=new_uuid(), ts=time.time(), message=message)
        self.enqueue(record)
        return record

    def append_metadata(self, model: str, cwd: str) -> MetadataRecord:
        record = MetadataRecord(
            uuid=new_uuid(),
            ts=time.time(),
            session_id=self.session_id,
            model=model,
            cwd=cwd,
        )
        self.enqueue(record)
        return record

    def append_compact_boundary(self, summary: str, pre_turn_count: int) -> CompactBoundaryRecord:
        record = CompactBoundaryRecord(
            uuid=new_uuid(), ts=time.time(), summary=summary, pre_turn_count=pre_turn_count
        )
        self.enqueue(record)
        return record

    async def flush(self) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        lines = "".join(_ADAPTER.dump_json(e).decode() + "\n" for e in batch)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            await f.write(lines)
