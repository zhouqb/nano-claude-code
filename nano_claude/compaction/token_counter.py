"""Estimating how full the context window currently is.

The most reliable signal is the input-token count the API reports for the most
recent request (``state.last_input_tokens``) — that is exactly the size of the
message history we just sent. Before any request has completed (e.g. right after
resuming a session), we fall back to a rough char-based estimate.
"""

from __future__ import annotations

import json
from typing import Any

# Rough bytes-per-token ratio for the fallback estimate.
_CHARS_PER_TOKEN = 4


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap, dependency-free estimate of the tokens in a message list."""
    chars = 0
    for message in messages:
        try:
            chars += len(json.dumps(message, default=str))
        except (TypeError, ValueError):
            chars += len(str(message))
    return chars // _CHARS_PER_TOKEN


def current_context_tokens(state: Any) -> int:
    """Best estimate of the current context size, in tokens."""
    if state.last_input_tokens:
        return state.last_input_tokens
    return estimate_message_tokens(state.messages)


def record_input_tokens(state: Any, chunk: Any) -> None:
    """Update ``state.last_input_tokens`` from a streaming chunk's usage field."""
    usage = getattr(chunk, "usage", None)
    if not usage:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens:
        state.last_input_tokens = prompt_tokens
