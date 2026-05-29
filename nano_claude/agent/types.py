"""Core types for the agent loop.

This is the Phase 1 subset of the types described in the plan. Fields that only
become meaningful in later phases (e.g. ``storage`` for compaction) are present
so the shape stays stable, but nothing in Phase 1 depends on them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nano_claude.permissions.modes import PermissionMode


class StopReason(StrEnum):
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    ABORTED = "aborted"
    ERROR = "error"


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
    context_window: int = 200_000  # overridden at startup from litellm.get_model_info()


@dataclass
class LoopState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    consecutive_compact_failures: int = 0
    # Populated in Phase 3 (session storage); unused in Phase 1.
    storage: Any | None = None


@dataclass
class LoopResult:
    reason: StopReason
    turn_count: int
    final_text: str
