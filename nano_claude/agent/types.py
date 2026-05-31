"""Core types for the agent loop.

This is the Phase 1 subset of the types described in the plan. Fields that only
become meaningful in later phases (e.g. ``storage`` for compaction) are present
so the shape stays stable, but nothing in Phase 1 depends on them.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nano_claude.permissions.modes import PermissionMode


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    ABORTED = "aborted"
    ERROR = "error"
    BLOCKED = "blocked"  # context full and auto-compact can't help; needs manual /compact


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    def merge(self, other: TokenUsage) -> None:
        """Roll another usage tally into this one (subagent cost → parent)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cache_read_tokens += other.cache_read_tokens

    def update_from_litellm(self, chunk: Any) -> None:
        """Accumulate usage from a LiteLLM streaming chunk's ``usage`` field.

        LiteLLM normalises usage to OpenAI's naming regardless of provider.
        """
        usage = getattr(chunk, "usage", None)
        if not usage:
            return
        self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        # Anthropic-style cache fields surface inside prompt_tokens_details.
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            self.cache_read_tokens += getattr(details, "cached_tokens", 0) or 0


@dataclass
class AgentConfig:
    model: str = "anthropic/claude-sonnet-4-6"
    max_turns: int = 50
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    auto_compact: bool = True
    context_collapse: bool = False  # Layer 4 (experimental); off by default
    # Layer 1 spill excerpt shape: "prefix" (head only) | "head_tail" (head + tail).
    tool_result_preview_format: str = "prefix"
    # Layer 3 (microcompact) time gate: clear old tool results only after this
    # many minutes since the last assistant message (cache presumed cold).
    microcompact_gap_minutes: int = 60
    context_window: int = 200_000  # overridden at startup from litellm.get_model_info()
    cwd: str = field(default_factory=os.getcwd)  # working directory for tool execution


@dataclass
class LoopState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    # Input tokens of the most recent request, as reported by the API. Used as
    # the current-context-size signal for compaction (Phase 4). 0 until known.
    last_input_tokens: int = 0
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    consecutive_compact_failures: int = 0
    # Populated in Phase 3 (session storage); unused in Phase 1.
    storage: Any | None = None
    # Layer 1 (budget) frozen-decision state; provisioned by the pipeline.
    budget: Any | None = None
    # Layer 4 (collapse) commit log; provisioned by the pipeline when enabled.
    collapse: Any | None = None
    # Wall-clock time (epoch seconds) of the last assistant message — the Layer 3
    # microcompact time gate. Set in the loop; seeded from the transcript on resume.
    last_assistant_at: float | None = None
    # Per-session file snapshots populated by Read and consumed by Edit/Write
    # stale-write guards.
    read_file_state: Any = field(default_factory=dict)


@dataclass
class LoopResult:
    reason: StopReason
    turn_count: int
    final_text: str
