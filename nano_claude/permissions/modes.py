"""Permission modes governing how tool-use decisions are resolved.

Only the enum is needed for Phase 1 (the skeleton has no tools yet); the
mode-transformation logic lands in Phase 2 alongside the permission manager.
"""

from enum import StrEnum


class PermissionMode(StrEnum):
    DEFAULT = "default"  # prompt user for every 'ask' decision
    ACCEPT_EDITS = "acceptEdits"  # auto-allow Read/Edit/Write; prompt for Bash
    BYPASS = "bypassPermissions"  # allow everything silently (CI / headless)
