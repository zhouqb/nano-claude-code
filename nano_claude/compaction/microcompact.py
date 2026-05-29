"""Layer 3 — Microcompact.

Surgically clears the *content* of OLD tool results while keeping the most
recent ``KEEP_RECENT`` of them verbatim. It never touches assistant text, user
messages, or the decision trail — only the bulky, stale bodies of past file
reads / command output. Mirrors Claude Code's ``microcompact`` (which clears
old tool_result content for a fixed set of compactable tools).

Only results from *compactable* tools are eligible — the ones whose output is
large and re-derivable (a file read, a command, a search), never a tool whose
result is itself a decision or side effect.

The clear is monotonic and deterministic: once a result scrolls out of the
recent window it is replaced with ``CLEARED_MESSAGE`` and never revived, so the
view stays prompt-cache stable turn over turn. Composes with Layer 1 (budget):
budget previews large *recent* results; microcompact fully clears *old* ones.

Simplification vs. Claude Code: CC also has a time-based trigger (clear when the
gap since the last assistant message means the cache is already cold). Nano's
in-memory messages carry no timestamp, so this is count-based only.
"""

from __future__ import annotations

# Tools whose results are large and re-derivable — safe to clear when stale.
COMPACTABLE_TOOLS = {"Bash", "Read", "Grep", "GlobTool", "Edit", "Write"}
CLEARED_MESSAGE = "[Old tool result content cleared]"
KEEP_RECENT = 6  # most-recent compactable results kept verbatim


def microcompact(messages: list[dict], *, keep_recent: int = KEEP_RECENT) -> list[dict]:
    """Clear old compactable tool-result bodies. Returns the same list if no-op."""
    # Resolve each tool result's originating tool via the assistant tool_calls.
    name_by_id: dict[str, str | None] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name_by_id[tc.get("id")] = (tc.get("function") or {}).get("name")

    compactable_ids = [
        msg.get("tool_call_id")
        for msg in messages
        if msg.get("role") == "tool"
        and name_by_id.get(msg.get("tool_call_id")) in COMPACTABLE_TOOLS
    ]
    # Floor keep_recent at 1: never clear literally everything.
    keep = set(compactable_ids[-max(1, keep_recent) :])
    clear = set(compactable_ids) - keep
    if not clear:
        return messages

    changed = False
    out: list[dict] = []
    for msg in messages:
        if (
            msg.get("role") == "tool"
            and msg.get("tool_call_id") in clear
            and msg.get("content") != CLEARED_MESSAGE
        ):
            out.append({**msg, "content": CLEARED_MESSAGE})
            changed = True
        else:
            out.append(msg)

    return out if changed else messages
