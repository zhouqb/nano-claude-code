"""Reading and writing memory files + the ``MEMORY.md`` index.

A memory directory holds one ``MEMORY.md`` index (one-line pointers, always
loaded into the system prompt) plus topic files (one fact each, with YAML
frontmatter ``name`` / ``description`` / ``type``). These helpers back the
``/remember`` and ``/forget`` commands and are exercised directly in tests; the
main agent also writes topic files inline via the ``Write``/``Edit`` tools.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from nano_claude.memory.paths import (
    ENTRYPOINT,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
)


def ensure_memory_dir(mdir: Path) -> None:
    """Create the memory directory (and parents). Idempotent."""
    mdir.mkdir(parents=True, exist_ok=True)


def read_entrypoint(mdir: Path) -> str:
    """Return the raw ``MEMORY.md`` content, or ``""`` if there is none."""
    path = mdir / ENTRYPOINT
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def truncate_entrypoint(raw: str) -> tuple[str, bool]:
    """Truncate ``MEMORY.md`` to the line and byte caps.

    Line-truncates first (a natural boundary), then byte-truncates at the last
    newline before the byte cap so we never cut mid-line. Appends a warning that
    names which cap fired. Returns ``(content, was_truncated)``.
    """
    trimmed = raw.strip()
    lines = trimmed.split("\n")
    line_count = len(lines)
    byte_count = len(trimmed)

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES
    if not was_line_truncated and not was_byte_truncated:
        return trimmed, False

    truncated = "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed
    if len(truncated) > MAX_ENTRYPOINT_BYTES:
        cut_at = truncated.rfind("\n", 0, MAX_ENTRYPOINT_BYTES)
        truncated = truncated[: cut_at if cut_at > 0 else MAX_ENTRYPOINT_BYTES]

    if was_byte_truncated and not was_line_truncated:
        reason = f"{byte_count} bytes (limit {MAX_ENTRYPOINT_BYTES}) — index entries are too long"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"

    warning = (
        f"\n\n> WARNING: {ENTRYPOINT} is {reason}. Only part of it was loaded. "
        "Keep index entries to one line under ~150 chars; move detail into topic files."
    )
    return truncated + warning, True


def _safe_filename(name: str) -> str:
    """Turn a memory name into a safe ``<slug>.md`` filename."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_.") or "memory"
    return slug if slug.endswith(".md") else f"{slug}.md"


def write_memory(
    mdir: Path,
    name: str,
    *,
    description: str,
    type: str,
    body: str,
) -> Path:
    """Write a topic file with frontmatter and return its path.

    Does not touch ``MEMORY.md`` — pointer maintenance is :func:`add_index_pointer`,
    matching the two-step save flow.
    """
    ensure_memory_dir(mdir)
    path = mdir / _safe_filename(name)
    post = frontmatter.Post(body.strip(), name=name, description=description, type=type)
    path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
    return path


def add_index_pointer(mdir: Path, title: str, filename: str, hook: str = "") -> None:
    """Append a one-line pointer to ``MEMORY.md`` (created if missing).

    Skips the write if a pointer to ``filename`` already exists, so re-saving an
    existing memory doesn't duplicate its index entry.
    """
    ensure_memory_dir(mdir)
    path = mdir / ENTRYPOINT
    existing = read_entrypoint(mdir)
    if f"({filename})" in existing:
        return
    suffix = f" — {hook}" if hook else ""
    line = f"- [{title}]({filename}){suffix}"
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{prefix}{line}\n")


def delete_memory(mdir: Path, filename: str) -> bool:
    """Delete a topic file and remove its ``MEMORY.md`` pointer line(s).

    Returns ``True`` if the topic file existed and was removed.
    """
    target = mdir / filename
    removed = False
    try:
        target.unlink()
        removed = True
    except OSError:
        pass

    entry = read_entrypoint(mdir)
    if entry and f"({filename})" in entry:
        kept = [ln for ln in entry.splitlines() if f"({filename})" not in ln]
        (mdir / ENTRYPOINT).write_text("\n".join(kept).strip() + "\n", encoding="utf-8")
    return removed
