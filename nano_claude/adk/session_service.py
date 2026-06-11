"""``JsonlSessionService``: ADK sessions persisted in nano's JSONL format.

Bridges ADK's ``BaseSessionService`` protocol onto the existing append-only
session storage (``session/storage.py``) and restore/crash-repair logic
(``session/restore.py``), so the on-disk format under
``~/.nano-claude/projects/<cwd>/`` stays byte-identical and ``--resume`` and
the JSONL tooling keep working.

Write-ownership split (the no-double-write rule):

- **This service persists model/tool events** — everything the agent
  produces during a turn flows through ``append_event``.
- **Callers persist user-authored and injected messages** — ``main.py``
  records the user prompt (and ``/clear``/hook system notes) directly via
  ``SessionStorage``, and the driver records memory/todo injections, exactly
  as before. ``append_event`` therefore *skips* user-authored events (the
  ``new_message`` echo the Runner appends), since the caller already wrote
  that message.

``create_session`` opens a lightweight transport session (empty event list;
the canonical history lives in ``LoopState.messages``). ``get_session`` for
an uncached id loads the JSONL from disk — records dedup'd, crash-repaired —
making persisted sessions consumable through plain ADK APIs.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import GetSessionConfig, ListSessionsResponse

from nano_claude.adk.convert import event_to_messages, messages_to_events
from nano_claude.session.restore import (
    list_sessions as list_session_infos,
)
from nano_claude.session.restore import (
    load_records,
    repair_messages,
    restore_messages,
)
from nano_claude.session.storage import (
    DEFAULT_ROOT,
    MessageRecord,
    SessionStorage,
    session_file,
)

_AGENT_NAME = "nano_claude"


class JsonlSessionService(BaseSessionService):
    """ADK session persistence backed by nano's per-project JSONL files.

    Contract: ``create_session``/``get_session`` return the *live* cached
    ``Session`` object (not a copy, unlike ADK's InMemorySessionService) —
    the driver's abort reconcile reads ``session.events`` to recover events
    that were persisted but never consumed, and relies on that identity.
    """

    def __init__(self, cwd: str, *, root: Path = DEFAULT_ROOT):
        self._cwd = cwd
        self._root = root
        self._sessions: dict[str, Session] = {}
        self._storages: dict[str, SessionStorage] = {}

    def adopt_storage(self, storage: SessionStorage) -> None:
        """Use an already-open ``SessionStorage`` (shared debounce queue)."""
        self._storages[storage.session_id] = storage

    def _storage_for(self, session_id: str) -> SessionStorage:
        storage = self._storages.get(session_id)
        if storage is None:
            storage = SessionStorage(session_file(self._cwd, session_id, self._root), session_id)
            self._storages[session_id] = storage
        return storage

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        if not session_id:
            from nano_claude.session.storage import new_session_id

            session_id = new_session_id()
        session = Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state=state or {},
            events=[],
            last_update_time=time.time(),
        )
        self._sessions[session_id] = session
        self._storage_for(session_id)
        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: GetSessionConfig | None = None,
    ) -> Session | None:
        cached = self._sessions.get(session_id)
        if cached is not None:
            return cached

        path = session_file(self._cwd, session_id, self._root)
        records = load_records(path)
        if not records:
            return None
        messages = repair_messages(restore_messages(records))
        # Timestamps/ids only exist for messages that came from records; the
        # repair step may have injected synthetic ones, so map by identity.
        record_meta = {
            id(r.message): (r.ts, r.uuid) for r in records if isinstance(r, MessageRecord)
        }
        timestamps = [record_meta.get(id(m), (0.0, ""))[0] for m in messages]
        event_ids = [record_meta.get(id(m), (0.0, ""))[1] for m in messages]
        session = Session(
            id=session_id,
            app_name=app_name,
            user_id=user_id,
            state={},
            events=messages_to_events(
                messages, agent_name=_AGENT_NAME, timestamps=timestamps, event_ids=event_ids
            ),
            last_update_time=path.stat().st_mtime if path.is_file() else time.time(),
        )
        self._sessions[session_id] = session
        return session

    async def append_event(self, session: Session, event: Event) -> Event:
        if event.partial:
            return event
        session.events.append(event)
        session.last_update_time = time.time()
        # The Runner echoes the caller's new_message as a user-authored event;
        # the caller (main.py / the driver's record()) already persisted it.
        if event.author == "user":
            return event
        storage = self._storage_for(session.id)
        for message in event_to_messages(event):
            storage.append_message(message)
        return event

    async def list_sessions(
        self, *, app_name: str, user_id: str | None = None
    ) -> ListSessionsResponse:
        sessions = [
            Session(
                id=info.session_id,
                app_name=app_name,
                user_id=user_id or "local",
                state={},
                events=[],
                last_update_time=info.mtime,
            )
            for info in list_session_infos(self._cwd, self._root)
        ]
        return ListSessionsResponse(sessions=sessions)

    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._storages.pop(session_id, None)
        path = session_file(self._cwd, session_id, self._root)
        path.unlink(missing_ok=True)
        outputs = path.parent / f"{session_id}-outputs"
        if outputs.is_dir():
            shutil.rmtree(outputs, ignore_errors=True)
