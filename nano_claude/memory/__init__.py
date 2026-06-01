"""Persistent, cross-session memory (Phase 8).

A curated, model-authored knowledge base of typed facts that should inform
*future* conversations — distinct from compaction (the current window) and
session storage (a verbatim transcript). Memory is scoped to the repo's
canonical git root, gated by :func:`is_memory_enabled`, and surfaced on demand
by a non-blocking per-turn relevance prefetch.
"""

from __future__ import annotations

from nano_claude.memory.paths import (
    ENTRYPOINT,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
    is_memory_enabled,
    memory_dir,
    validate_memory_path,
)
from nano_claude.memory.prompt import build_memory_section
from nano_claude.memory.recall import (
    MemoryPrefetch,
    MemorySession,
    find_relevant_memories,
    surface_memory_attachment,
)
from nano_claude.memory.store import (
    add_index_pointer,
    delete_memory,
    ensure_memory_dir,
    read_entrypoint,
    truncate_entrypoint,
    write_memory,
)

__all__ = [
    "ENTRYPOINT",
    "MAX_ENTRYPOINT_BYTES",
    "MAX_ENTRYPOINT_LINES",
    "MemoryPrefetch",
    "MemorySession",
    "add_index_pointer",
    "build_memory_section",
    "delete_memory",
    "ensure_memory_dir",
    "find_relevant_memories",
    "is_memory_enabled",
    "memory_dir",
    "read_entrypoint",
    "surface_memory_attachment",
    "truncate_entrypoint",
    "validate_memory_path",
    "write_memory",
]
