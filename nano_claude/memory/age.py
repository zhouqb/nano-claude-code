"""Memory freshness helpers.

Models reason about staleness from human phrasing ("47 days ago") far better
than from a raw timestamp, and a stale file:line citation sounds *more*
authoritative, not less — so recalled memories carry an explicit age and, when
old, a caveat. Mirrors ``src/memdir/memoryAge.ts``.
"""

from __future__ import annotations

import time

_DAY_SECONDS = 86_400


def memory_age_days(mtime: float, *, now: float | None = None) -> int:
    """Whole days since ``mtime``; clamps negative (clock skew) to 0."""
    now = time.time() if now is None else now
    return max(0, int((now - mtime) // _DAY_SECONDS))


def memory_age(mtime: float, *, now: float | None = None) -> str:
    """Human-readable age: ``today`` / ``yesterday`` / ``N days ago``."""
    days = memory_age_days(mtime, now=now)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def memory_freshness_text(mtime: float, *, now: float | None = None) -> str:
    """Staleness caveat for memories older than a day; ``""`` for fresh ones."""
    days = memory_age_days(mtime, now=now)
    if days <= 1:
        return ""
    return (
        f"This memory is {days} days old. Memories are point-in-time observations, "
        "not live state — claims about code or file:line citations may be outdated. "
        "Verify against the current code before relying on it."
    )
