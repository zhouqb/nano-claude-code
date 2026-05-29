"""Layer 2 — Snip.

A cheap, structural prune of *zombie* messages — entries that carry no forward
value for the model. No API call, no summarization: just drop the dead weight.
Mirrors Claude Code's ``snipCompact`` ("remove zombie messages and stale
markers"), which runs before microcompact/collapse/auto-compact.

What nano considers a zombie:

  * a **fully-interrupted tool turn** — an assistant message with ``tool_calls``
    where *every* matching tool result is the synthetic ``"[Interrupted]"``
    (left behind by crash recovery). The assistant message and those results
    are removed together, so tool-call/result pairing stays API-valid.
  * a standalone **``[Interrupted]`` assistant marker** (no tool_calls).
  * an **empty assistant message** (no content, no tool_calls).

A partially-interrupted turn (some real results, some interrupted) is kept — we
only drop turns that are entirely dead.

Snip runs every turn on the derived view; ``/snip`` applies it to the canonical
store to prune scrollback too. Because nano's context signal is the
API-reported size of the *already-snipped* send (``last_input_tokens``), the
savings are reflected automatically next turn — no token plumbing needed (unlike
CC, whose count comes from a protected-tail message that survives snip).
"""

from __future__ import annotations

from dataclasses import dataclass

from nano_claude.compaction.token_counter import estimate_message_tokens

INTERRUPTED = "[Interrupted]"


@dataclass
class SnipResult:
    messages: list[dict]
    removed: int  # number of messages dropped
    tokens_freed: int  # rough estimate of what was dropped (informational)


def _is_empty_assistant(msg: dict) -> bool:
    if msg.get("role") != "assistant" or msg.get("tool_calls"):
        return False
    content = msg.get("content")
    return content is None or (isinstance(content, str) and not content.strip())


def _is_interrupted_marker(msg: dict) -> bool:
    return (
        msg.get("role") == "assistant"
        and not msg.get("tool_calls")
        and msg.get("content") == INTERRUPTED
    )


def snip_messages(messages: list[dict]) -> SnipResult:
    """Drop zombie messages. Returns the same list object when nothing is removed."""
    # Map each tool result's call id to its content, to detect dead tool turns.
    result_by_id = {
        m.get("tool_call_id"): m.get("content") for m in messages if m.get("role") == "tool"
    }

    drop_messages: set[int] = set()  # indices to remove
    drop_tool_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = [tc.get("id") for tc in msg["tool_calls"]]
            # Dead turn: every call was answered, and all answers are [Interrupted].
            if ids and all(result_by_id.get(tid) == INTERRUPTED for tid in ids):
                drop_messages.add(i)
                drop_tool_ids.update(ids)
        elif _is_empty_assistant(msg) or _is_interrupted_marker(msg):
            drop_messages.add(i)

    if not drop_messages and not drop_tool_ids:
        return SnipResult(messages, 0, 0)

    kept: list[dict] = []
    removed: list[dict] = []
    for i, msg in enumerate(messages):
        if i in drop_messages or (
            msg.get("role") == "tool" and msg.get("tool_call_id") in drop_tool_ids
        ):
            removed.append(msg)
        else:
            kept.append(msg)

    return SnipResult(kept, len(removed), estimate_message_tokens(removed))
