"""Layer 3 — Microcompact.

Surgically clears the *content* of OLD tool results while keeping the most
recent ``KEEP_RECENT`` of them verbatim. It never touches assistant text, user
messages, or the decision trail — only the bulky, stale bodies of past file
reads / command output. Mirrors Claude Code's ``microcompact`` (which clears
old tool_result content for a fixed set of compactable tools).

Only results from *compactable* tools are eligible — the ones whose output is
large and re-derivable (a file read, a command, a search), never a tool whose
result is itself a decision or side effect.

**Time-gated trigger.** Clearing only happens when the gap since the last
assistant message exceeds ``gap_threshold_minutes`` — i.e. the server-side
prompt cache (≈1h TTL) has almost certainly expired, so the prefix will be
rewritten anyway and clearing old results now is *free*. Below the threshold
this is a no-op: mutating mid-prefix content while the cache is still warm would
force a cache miss on everything after the edit, which is exactly what we avoid.
This mirrors Claude Code's ``maybeTimeBasedMicrocompact`` (gap default 60 min).

The clear is monotonic and deterministic: once cleared a result is never revived
(the ``!= CLEARED_MESSAGE`` guard), so re-runs are stable. Composes with Layer 1
(budget): budget previews large *recent* results; microcompact clears *old* ones.

Simplification vs. Claude Code: CC's other path, ``cachedMicrocompactPath``,
trims old results *during* an active (warm-cache) session via the cache-editing
API without busting the prefix. That API isn't exposed by litellm, so nano only
has the time-based (cold-cache) path — meaning in an active session microcompact
does nothing and in-session pressure is handled by auto-compact (Layer 5).
"""

from __future__ import annotations

# Tools whose results are large and re-derivable — safe to clear when stale.
COMPACTABLE_TOOLS = {"Bash", "Read", "Grep", "GlobTool", "Edit", "Write"}
CLEARED_MESSAGE = "[Old tool result content cleared]"
KEEP_RECENT = 5  # most-recent compactable results kept verbatim (CC default)
# Clear only once the gap since the last assistant message exceeds this, so the
# cache is already cold. 60 min: past the server's ~1h TTL for every user.
GAP_THRESHOLD_MINUTES = 60


def microcompact(
    messages: list[dict],
    *,
    keep_recent: int = KEEP_RECENT,
    gap_minutes: float | None = None,
    gap_threshold_minutes: float = GAP_THRESHOLD_MINUTES,
) -> list[dict]:
    """Clear old compactable tool-result bodies. Returns the same list if no-op.

    ``gap_minutes`` is the wall-clock time since the last assistant message (the
    pipeline computes it from ``LoopState.last_assistant_at``). Clearing is gated
    on it: ``None`` (no prior assistant / unknown) or below the threshold → no-op,
    so we never bust a warm cache. See the module docstring.
    """
    if gap_minutes is None or gap_minutes < gap_threshold_minutes:
        return messages
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
