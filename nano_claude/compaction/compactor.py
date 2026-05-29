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

# Structured summary template (mirrors Claude Code's compact/prompt.ts). The
# summary REPLACES the earlier history, so it must be thorough and
# self-contained; the section structure is what keeps the model from dropping
# load-bearing detail (user intent, file edits, errors, the in-flight task).
SUMMARY_PROMPT = """\
Your task is to create a detailed summary of the conversation so far. This \
summary will REPLACE the earlier conversation history, so capture everything \
needed to continue the work seamlessly — be thorough and self-contained.

Structure your summary with these sections:

1. Primary Request and Intent: What the user asked for, in their own framing, \
including any explicit constraints or preferences.
2. Key Technical Concepts: Technologies, frameworks, and design decisions in play.
3. Files and Code Sections: Specific files examined, created, or modified. \
Include important code snippets and why each file matters.
4. Errors and Fixes: Errors encountered and how they were resolved, plus any \
user feedback on them.
5. Problem Solving: Problems solved and ongoing troubleshooting.
6. All User Messages: List every non-tool-result user message verbatim — these \
are critical for tracking intent and feedback.
7. Pending Tasks: Anything explicitly requested that is not yet done.
8. Current Work: Precisely what was being worked on immediately before this \
summary, with file names and code where relevant.
9. Next Step (optional): The next action, only if it directly continues the \
most recent task. Quote the relevant request to avoid drift."""


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
