"""Scan a memory directory into a header manifest for relevance selection.

Reads only each file's frontmatter (name / description / type) plus its mtime —
cheap enough to run every turn. The manifest is what the relevance side-query
sees; full file contents are read only for the handful that get selected.
Mirrors ``src/memdir/memoryScan.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import frontmatter

from nano_claude.memory.paths import ENTRYPOINT

MAX_MEMORY_FILES = 200
# Only the leading lines of each file are read to extract frontmatter during a
# scan — this runs every turn (recall) and again for extraction, so we never
# slurp a large accidental memory file here. Full content is read only for the
# few selected files, in surface_memory_attachment() (with its own caps).
FRONTMATTER_MAX_LINES = 30


@dataclass(frozen=True)
class MemoryHeader:
    filename: str  # path relative to the memory dir (so subdirs stay distinct)
    path: Path
    mtime: float
    description: str | None
    type: str | None


def _read_frontmatter_meta(path: Path) -> dict:
    """Parse frontmatter from only the first ``FRONTMATTER_MAX_LINES`` of a file.

    A memory's frontmatter block is tiny; reading a bounded prefix keeps scans
    cheap regardless of body size. If the block isn't closed within the prefix
    (or there is none), the metadata is simply empty.
    """
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= FRONTMATTER_MAX_LINES:
                break
            lines.append(line)
    try:
        return frontmatter.loads("".join(lines)).metadata
    except Exception:  # noqa: BLE001 - malformed frontmatter → treat as none
        return {}


def scan_memory_files(mdir: Path) -> list[MemoryHeader]:
    """Return frontmatter headers for every ``*.md`` (except ``MEMORY.md``).

    Sorted newest-first and capped at :data:`MAX_MEMORY_FILES`. Only a bounded
    frontmatter prefix is read per file. Unreadable files are skipped, never fatal.
    """
    if not mdir.is_dir():
        return []

    headers: list[MemoryHeader] = []
    for path in mdir.rglob("*.md"):
        if path.name == ENTRYPOINT or not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
            meta = _read_frontmatter_meta(path)
        except OSError:  # one bad file must not break recall
            continue
        desc = meta.get("description")
        mtype = meta.get("type")
        headers.append(
            MemoryHeader(
                filename=str(path.relative_to(mdir)),
                path=path,
                mtime=mtime,
                description=str(desc) if desc else None,
                type=str(mtype) if mtype else None,
            )
        )

    headers.sort(key=lambda h: h.mtime, reverse=True)
    return headers[:MAX_MEMORY_FILES]


def format_manifest(headers: list[MemoryHeader]) -> str:
    """One line per file: ``- [type] filename (iso-ts): description``."""
    lines = []
    for h in headers:
        tag = f"[{h.type}] " if h.type else ""
        ts = datetime.fromtimestamp(h.mtime).isoformat(timespec="seconds")
        desc = f": {h.description}" if h.description else ""
        lines.append(f"- {tag}{h.filename} ({ts}){desc}")
    return "\n".join(lines)
