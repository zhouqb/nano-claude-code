"""Tests for memory recall: scan, relevance select, surfacing, and prefetch."""

from __future__ import annotations

import asyncio
import json
import time

import litellm

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.memory.age import memory_age, memory_age_days, memory_freshness_text
from nano_claude.memory.recall import (
    MAX_SESSION_BYTES,
    MemorySession,
    find_relevant_memories,
    surface_memory_attachment,
)
from nano_claude.memory.scan import format_manifest, scan_memory_files
from nano_claude.memory.store import write_memory
from nano_claude.permissions.settings import Settings
from tests.conftest import make_acompletion, text_chunk, usage_chunk


def _select_acompletion(selected):
    """Fake acompletion returning a JSON {selected: [...]} relevance response."""

    async def _fn(*args, **kwargs):
        payload = json.dumps({"selected": selected})
        msg = type("M", (), {"content": payload})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()

    return _fn


# --- age --------------------------------------------------------------------


def test_age_phrasing():
    now = time.time()
    assert memory_age(now, now=now) == "today"
    assert memory_age(now - 86_400, now=now) == "yesterday"
    assert memory_age(now - 5 * 86_400, now=now) == "5 days ago"
    assert memory_age_days(now + 100, now=now) == 0  # clock skew clamps to 0


def test_freshness_text_only_for_old():
    now = time.time()
    assert memory_freshness_text(now, now=now) == ""
    assert "days old" in memory_freshness_text(now - 10 * 86_400, now=now)


# --- scan + manifest --------------------------------------------------------


def test_scan_reads_headers_and_skips_entrypoint(tmp_path):
    write_memory(tmp_path, "a", description="about A", type="user", body="x")
    (tmp_path / "MEMORY.md").write_text("- [A](a.md)\n")
    headers = scan_memory_files(tmp_path)
    assert [h.filename for h in headers] == ["a.md"]
    assert headers[0].type == "user"
    assert headers[0].description == "about A"


def test_scan_sorts_newest_first(tmp_path):
    write_memory(tmp_path, "old", description="o", type="user", body="x")
    time.sleep(0.01)
    write_memory(tmp_path, "new", description="n", type="user", body="x")
    headers = scan_memory_files(tmp_path)
    assert headers[0].filename == "new.md"


def test_manifest_format(tmp_path):
    write_memory(tmp_path, "a", description="desc", type="project", body="x")
    line = format_manifest(scan_memory_files(tmp_path))
    assert "[project]" in line and "a.md" in line and "desc" in line


# --- find_relevant_memories -------------------------------------------------


async def test_find_returns_selected_paths(tmp_path, monkeypatch):
    write_memory(tmp_path, "uv", description="uses uv", type="user", body="x")
    write_memory(tmp_path, "tz", description="timezone", type="user", body="x")
    monkeypatch.setattr(litellm, "acompletion", _select_acompletion(["uv.md"]))
    paths = await find_relevant_memories("how do deps work", tmp_path, recall_model="m")
    assert [p.name for p in paths] == ["uv.md"]


async def test_find_excludes_already_surfaced(tmp_path, monkeypatch):
    write_memory(tmp_path, "uv", description="uses uv", type="user", body="x")
    monkeypatch.setattr(litellm, "acompletion", _select_acompletion(["uv.md"]))
    surfaced = frozenset({str(tmp_path / "uv.md")})
    paths = await find_relevant_memories(
        "deps", tmp_path, recall_model="m", already_surfaced=surfaced
    )
    assert paths == []


async def test_find_swallows_side_query_errors(tmp_path, monkeypatch):
    write_memory(tmp_path, "uv", description="uses uv", type="user", body="x")

    async def _boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(litellm, "acompletion", _boom)
    assert await find_relevant_memories("deps", tmp_path, recall_model="m") == []


# --- surfacing --------------------------------------------------------------


def test_surface_wraps_in_reminder_with_age(tmp_path):
    path = write_memory(tmp_path, "uv", description="d", type="user", body="Use uv.")
    msg = surface_memory_attachment(path)
    assert msg["role"] == "user"
    assert "<system-reminder>" in msg["content"]
    assert "Use uv." in msg["content"]
    assert "saved today" in msg["content"]


def test_surface_caps_long_file_and_notes_truncation(tmp_path):
    big = "line\n" * 500
    path = write_memory(tmp_path, "big", description="d", type="project", body=big)
    msg = surface_memory_attachment(path)
    assert "Truncated" in msg["content"]


# --- MemorySession / prefetch ------------------------------------------------


async def test_prefetch_dedups_and_enforces_session_cap(tmp_path, monkeypatch):
    write_memory(tmp_path, "a", description="d", type="user", body="A body")
    monkeypatch.setattr(litellm, "acompletion", _select_acompletion(["a.md"]))
    session = MemorySession(mdir=tmp_path, recall_model="m")

    pf = session.start("q")
    await pf.task
    first = pf.drain_if_ready()
    assert len(first) == 1
    assert str(tmp_path / "a.md") in session.surfaced
    # Idempotent: a second drain of the same (consumed) prefetch yields nothing.
    assert pf.drain_if_ready() == []

    # A later turn won't re-surface the same file.
    pf2 = session.start("q again")
    await pf2.task
    assert pf2.drain_if_ready() == []


async def test_prefetch_not_ready_yields_nothing(tmp_path):
    session = MemorySession(mdir=tmp_path, recall_model="m")

    async def _slow(*a, **k):
        await asyncio.sleep(10)
        return []

    pf = session.start("q")
    pf.task = asyncio.create_task(_slow())
    try:
        assert pf.drain_if_ready() == []  # not done yet → zero-wait, nothing surfaced
        assert pf.consumed is False
    finally:
        pf.cancel()


async def test_session_cap_blocks_further_surfacing(tmp_path, monkeypatch):
    write_memory(tmp_path, "a", description="d", type="user", body="A")
    monkeypatch.setattr(litellm, "acompletion", _select_acompletion(["a.md"]))
    session = MemorySession(mdir=tmp_path, recall_model="m", bytes_used=MAX_SESSION_BYTES)
    pf = session.start("q")
    await pf.task
    assert pf.drain_if_ready() == []  # over the ceiling → nothing


# --- loop integration -------------------------------------------------------


async def test_loop_consumes_ready_prefetch(tmp_path, monkeypatch):
    """A settled prefetch is drained and its memory injected into the transcript."""
    write_memory(tmp_path, "uv", description="uses uv", type="user", body="Prefer uv.")
    monkeypatch.setattr(litellm, "acompletion", _select_acompletion(["uv.md"]))
    session = MemorySession(mdir=tmp_path, recall_model="m")
    pf = session.start("q")
    await pf.task  # ensure settled before the loop runs

    # Main model just answers (no tools).
    monkeypatch.setattr(
        litellm, "acompletion", make_acompletion([text_chunk("ok"), usage_chunk(1, 1)])
    )

    state = LoopState(messages=[{"role": "user", "content": "q"}])
    result = await query_loop(state, AgentConfig(), settings=Settings(), memory_prefetch=pf)

    assert result.reason is StopReason.COMPLETED
    injected = [m for m in state.messages if "<system-reminder>" in str(m.get("content", ""))]
    assert injected and "Prefer uv." in injected[0]["content"]


async def test_loop_skips_unsettled_prefetch(tmp_path, monkeypatch):
    """A prefetch that never settles must not block or inject anything."""
    session = MemorySession(mdir=tmp_path, recall_model="m")

    async def _never(*a, **k):
        await asyncio.sleep(10)
        return []

    pf = session.start("q")
    pf.task = asyncio.create_task(_never())
    monkeypatch.setattr(
        litellm, "acompletion", make_acompletion([text_chunk("ok"), usage_chunk(1, 1)])
    )
    state = LoopState(messages=[{"role": "user", "content": "q"}])
    try:
        result = await query_loop(state, AgentConfig(), settings=Settings(), memory_prefetch=pf)
    finally:
        pf.cancel()

    assert result.reason is StopReason.COMPLETED
    assert not any("<system-reminder>" in str(m.get("content", "")) for m in state.messages)
