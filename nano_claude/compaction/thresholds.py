"""Context-window thresholds.

The model's context window varies (populated at startup from
``litellm.get_model_info``). Thresholds are expressed as fixed token buffers
subtracted from that window, so the same constants work for a 64k or a 1M model.
"""

from __future__ import annotations

# Fixed buffers (tokens reserved before the hard context limit).
WARN_BUFFER = 20_000  # show a warning to the user
AUTO_COMPACT_BUFFER = 13_000  # auto-compaction triggers here
BLOCK_BUFFER = 3_000  # context effectively full; force manual /compact
SUMMARY_RESERVE = 8_000  # tokens reserved for the generated summary

# Context collapse (Layer 4) starts committing at this fraction of the window —
# below the auto-compact threshold, so collapse gets first crack at the headroom.
COLLAPSE_COMMIT_FRACTION = 0.90

# Give up auto-compacting after this many consecutive failures (circuit breaker).
MAX_CONSECUTIVE_COMPACT_FAILURES = 3


def warn_threshold(context_window: int) -> int:
    return context_window - WARN_BUFFER


def auto_compact_threshold(context_window: int) -> int:
    return context_window - AUTO_COMPACT_BUFFER


def block_threshold(context_window: int) -> int:
    return context_window - BLOCK_BUFFER


def collapse_commit_threshold(context_window: int) -> int:
    return int(context_window * COLLAPSE_COMMIT_FRACTION)
