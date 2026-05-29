"""Conversation compaction: summarize the history and replace it.

On success the message list becomes ``[system?, summary, *recent_tail]`` where
the recent tail is kept API-valid (no orphan tool messages, no dangling tool
calls). On failure the failure counter is bumped for the circuit breaker.
"""

from __future__ import annotations

import litellm

from nano_claude.agent.types import AgentConfig, LoopState
from nano_claude.session.restore import repair_messages

# How many of the most recent (non-system) messages to keep verbatim.
RECENT_MESSAGES_KEPT = 6

SUMMARY_PROMPT = (
    "Summarize the conversation so far. Preserve all tool outputs, decisions "
    "made, files created or changed, important code, and any unresolved tasks or "
    "next steps. This summary will REPLACE the earlier conversation history, so "
    "be thorough and self-contained."
)


async def _summarize(state: LoopState, config: AgentConfig) -> str:
    """Ask the model to summarize the conversation (streamed, then joined)."""
    messages = [*state.messages, {"role": "user", "content": SUMMARY_PROMPT}]
    response = await litellm.acompletion(
        model=config.model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    async for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        content = getattr(choices[0].delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts).strip()


def _safe_recent_tail(messages: list[dict]) -> list[dict]:
    """Keep the last few non-system messages as a valid, self-contained tail."""
    non_system = [m for m in messages if m.get("role") != "system"]
    tail = non_system[-RECENT_MESSAGES_KEPT:]
    # Drop leading orphan tool results whose assistant call isn't in the tail.
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)
    # Repair any dangling tool_calls left at the end of the tail.
    return repair_messages(tail)


async def compact_conversation(state: LoopState, config: AgentConfig) -> bool:
    """Compact ``state.messages`` in place. Returns True on success."""
    pre_turn_count = state.turn_count
    try:
        summary = await _summarize(state, config)
    except Exception:  # noqa: BLE001 - any failure feeds the circuit breaker
        state.consecutive_compact_failures += 1
        return False

    if not summary:
        state.consecutive_compact_failures += 1
        return False

    system = next((m for m in state.messages if m.get("role") == "system"), None)
    tail = _safe_recent_tail(state.messages)

    new_messages: list[dict] = []
    if system is not None:
        new_messages.append(system)
    new_messages.append(
        {"role": "user", "content": f"[Summary of earlier conversation]\n\n{summary}"}
    )
    new_messages.extend(tail)

    state.messages = new_messages
    state.last_input_tokens = 0
    state.consecutive_compact_failures = 0

    if state.storage is not None:
        state.storage.append_compact_boundary(summary=summary, pre_turn_count=pre_turn_count)
    return True
