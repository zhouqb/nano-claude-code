"""Tests for memory file I/O, the MEMORY.md index, and truncation."""

from __future__ import annotations

import frontmatter

from nano_claude.memory.paths import ENTRYPOINT, MAX_ENTRYPOINT_LINES
from nano_claude.memory.store import (
    add_index_pointer,
    delete_memory,
    read_entrypoint,
    truncate_entrypoint,
    write_memory,
)

# --- truncate_entrypoint ----------------------------------------------------


def test_under_cap_is_unchanged():
    raw = "- [A](a.md)\n- [B](b.md)\n"
    content, truncated = truncate_entrypoint(raw)
    assert truncated is False
    assert content == raw.strip()


def test_over_line_cap_truncates_with_warning():
    raw = "\n".join(f"- [m{i}](m{i}.md)" for i in range(MAX_ENTRYPOINT_LINES + 50))
    content, truncated = truncate_entrypoint(raw)
    assert truncated is True
    assert "WARNING" in content
    # The kept body (before the warning) is capped at the line limit.
    body = content.split("> WARNING", 1)[0].strip()
    assert len(body.splitlines()) == MAX_ENTRYPOINT_LINES


def test_over_byte_cap_truncates_on_one_line():
    raw = "- [big](big.md) " + "x" * 30_000
    content, truncated = truncate_entrypoint(raw)
    assert truncated is True
    assert "WARNING" in content


# --- write_memory + frontmatter round-trip ----------------------------------


def test_write_memory_round_trips(tmp_path):
    path = write_memory(
        tmp_path,
        "user-prefers-uv",
        description="Installs deps with uv",
        type="user",
        body="The user uses `uv`, never pip.",
    )
    assert path.exists()
    post = frontmatter.load(str(path))
    assert post["name"] == "user-prefers-uv"
    assert post["type"] == "user"
    assert post["description"] == "Installs deps with uv"
    assert "uv" in post.content


def test_write_memory_sanitizes_filename(tmp_path):
    path = write_memory(tmp_path, "weird / name!!", description="d", type="project", body="b")
    assert path.suffix == ".md"
    assert "/" not in path.name


# --- index pointers ---------------------------------------------------------


def test_add_index_pointer_creates_and_appends(tmp_path):
    add_index_pointer(tmp_path, "User prefers uv", "user-prefers-uv.md", "uv not pip")
    text = read_entrypoint(tmp_path)
    assert "- [User prefers uv](user-prefers-uv.md) — uv not pip" in text


def test_add_index_pointer_dedups_by_filename(tmp_path):
    add_index_pointer(tmp_path, "A", "a.md")
    add_index_pointer(tmp_path, "A again", "a.md")
    text = read_entrypoint(tmp_path)
    assert text.count("(a.md)") == 1


# --- delete -----------------------------------------------------------------


def test_delete_memory_removes_file_and_pointer(tmp_path):
    write_memory(tmp_path, "temp", description="d", type="project", body="b")
    add_index_pointer(tmp_path, "Temp", "temp.md")
    assert (tmp_path / "temp.md").exists()

    removed = delete_memory(tmp_path, "temp.md")
    assert removed is True
    assert not (tmp_path / "temp.md").exists()
    assert "(temp.md)" not in read_entrypoint(tmp_path)


def test_delete_missing_file_returns_false(tmp_path):
    (tmp_path / ENTRYPOINT).write_text("- [x](nope.md)\n")
    assert delete_memory(tmp_path, "nope.md") is False
