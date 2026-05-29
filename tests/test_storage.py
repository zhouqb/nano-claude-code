"""Tests for the JSONL session storage write queue."""

from __future__ import annotations

from nano_claude.session.storage import (
    MessageRecord,
    MetadataRecord,
    SessionStorage,
    new_session_id,
    parse_record,
    project_dir,
    sanitize_cwd,
    session_file,
)


def test_session_id_is_unique():
    assert new_session_id() != new_session_id()


def test_sanitize_and_paths(tmp_path):
    assert sanitize_cwd("/Users/me/proj") == "Users-me-proj"
    pdir = project_dir("/Users/me/proj", root=tmp_path)
    assert pdir == tmp_path / "projects" / "Users-me-proj"
    sfile = session_file("/Users/me/proj", "sid", root=tmp_path)
    assert sfile.name == "sid.jsonl"


def test_parse_record_round_trip():
    rec = MessageRecord(uuid="u1", ts=1.0, message={"role": "user", "content": "hi"})
    line = rec.model_dump_json()
    parsed = parse_record(line)
    assert isinstance(parsed, MessageRecord)
    assert parsed.message == {"role": "user", "content": "hi"}


def test_parse_record_discriminates_metadata():
    rec = MetadataRecord(uuid="u", ts=1.0, session_id="s", model="m", cwd="/c")
    parsed = parse_record(rec.model_dump_json())
    assert isinstance(parsed, MetadataRecord)


def test_parse_record_handles_garbage():
    assert parse_record("") is None
    assert parse_record("not json") is None
    assert parse_record('{"type": "unknown"}') is None


async def test_flush_writes_jsonl(tmp_path):
    path = tmp_path / "s.jsonl"
    storage = SessionStorage(path, "sid")
    storage.append_metadata(model="gpt", cwd="/c")
    storage.append_message({"role": "user", "content": "hello"})
    storage.append_message({"role": "assistant", "content": "hi"})
    await storage.flush()

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    records = [parse_record(line) for line in lines]
    assert isinstance(records[0], MetadataRecord)
    assert records[1].message["content"] == "hello"


async def test_flush_is_idempotent_when_empty(tmp_path):
    path = tmp_path / "s.jsonl"
    storage = SessionStorage(path, "sid")
    await storage.flush()  # nothing pending
    assert not path.exists()


async def test_appends_accumulate_across_flushes(tmp_path):
    path = tmp_path / "s.jsonl"
    storage = SessionStorage(path, "sid")
    storage.append_message({"role": "user", "content": "one"})
    await storage.flush()
    storage.append_message({"role": "user", "content": "two"})
    await storage.flush()
    assert len(path.read_text().strip().splitlines()) == 2


async def test_message_record_preserves_tool_calls(tmp_path):
    path = tmp_path / "s.jsonl"
    storage = SessionStorage(path, "sid")
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
        ],
    }
    storage.append_message(msg)
    await storage.flush()
    parsed = parse_record(path.read_text().strip())
    assert parsed.message["tool_calls"][0]["id"] == "call_1"
