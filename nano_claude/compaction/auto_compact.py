"""Threshold checks that drive the compaction pipeline."""

from __future__ import annotations

from nano_claude.agent.types import AgentConfig, LoopState
from nano_claude.compaction.thresholds import (
    MAX_CONSECUTIVE_COMPACT_FAILURES,
    auto_compact_threshold,
    block_threshold,
    warn_threshold,
)
from nano_claude.compaction.token_counter import current_context_tokens


def circuit_broken(state: LoopState) -> bool:
    return state.consecutive_compact_failures >= MAX_CONSECUTIVE_COMPACT_FAILURES


def should_auto_compact(state: LoopState, config: AgentConfig) -> bool:
    if not config.auto_compact or circuit_broken(state):
        return False
    return current_context_tokens(state) >= auto_compact_threshold(config.context_window)


def should_warn(state: LoopState, config: AgentConfig) -> bool:
    return current_context_tokens(state) >= warn_threshold(config.context_window)


def should_block(state: LoopState, config: AgentConfig) -> bool:
    return current_context_tokens(state) >= block_threshold(config.context_window)
