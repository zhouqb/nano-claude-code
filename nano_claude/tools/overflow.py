"""Spill truncated tool output to disk so nothing is lost permanently.

When a tool truncates its output, it calls :func:`save_overflow` to persist the
*full* output to a file and :func:`truncation_note` to append a pointer to it.
The pointer names the path and tells the model to use the Read tool (which
takes any absolute path and paginates) to view the rest.

Spilling is best-effort: any failure returns ``None`` and the tool falls back to
a plain truncation marker — it must never break the tool call itself.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

from nano_claude.tools.base import ToolContext

# Best-effort cleanup: prune spill files older than this on each save.
SPILL_TTL_S = 7 * 24 * 3600  # 7 days


def _resolve_dir(output_dir: Path | None) -> Path:
    """The given outputs dir if set, else a shared temp dir."""
    if output_dir is not None:
        return output_dir
    return Path(tempfile.gettempdir()) / "nano-claude-outputs"


def _prune_old(directory: Path) -> None:
    cutoff = time.time() - SPILL_TTL_S
    try:
        for f in directory.glob("*.txt"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def save_overflow_to(content: str, name: str, output_dir: Path | None) -> Path | None:
    """Write ``content`` to ``<output_dir>/<name>.txt``; return its path or None.

    The filename is taken verbatim (no timestamp/uuid), so callers that pass a
    stable ``name`` get a deterministic, idempotent path — re-spilling the same
    logical content overwrites in place rather than leaking a new file.
    """
    try:
        directory = _resolve_dir(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        _prune_old(directory)
        path = directory / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
        return path
    except OSError:
        return None


def save_overflow(content: str, tool_name: str, context: ToolContext) -> Path | None:
    """Write ``content`` to a uniquely-named spill file; return its path or None."""
    unique = f"{tool_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    return save_overflow_to(content, unique, context.output_dir)


def truncation_note(path: Path | None, *, shown: int, total: int, unit: str = "bytes") -> str:
    """The suffix appended to truncated output; degrades if the spill failed."""
    base = f"\n... (output truncated: showing {shown} of {total} {unit}."
    if path is None:
        return base + ")"
    return base + f" Full output saved to {path} — use the Read tool to view the rest.)"
