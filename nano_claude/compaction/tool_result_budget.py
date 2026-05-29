"""Layer 1 — Budget Reduction.

Caps the size of tool results in the conversation. A result larger than
``PER_RESULT_CHAR_CAP`` is spilled to disk and replaced in the *view* with a
short preview that points the model at the saved file (which it can Read back).
The canonical ``state.messages`` keeps the full content for storage/scrollback.

The decisive property is **frozen, cache-stable decisions** (mirrors Claude
Code's ``enforceToolResultBudget``): each result's fate is keyed by
``tool_call_id`` and never changes once made —

  * already-replaced → re-apply the exact same preview string every turn
    (a dict lookup, byte-identical, so the prompt-cache prefix is preserved);
  * seen-but-not-replaced → never replaced later.

Spill paths are derived deterministically from the ``tool_call_id``, so the
same result yields the same preview even across ``--resume`` (when the in-memory
state starts empty and the decision is re-derived) — no leaked files, no churn.

Simplification vs. Claude Code: there the budget is a per-*message* aggregate
(many tool_results share one user message). In nano each tool result is its own
``role: "tool"`` message, so the aggregate reduces to a per-result cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nano_claude.tools.overflow import save_overflow_to

# Results longer than this many characters get spilled + previewed (~4k tokens).
PER_RESULT_CHAR_CAP = 16_000
# How much of the head of the result to keep inline in the preview.
PREVIEW_CHARS = 500


@dataclass
class ContentReplacementState:
    """Per-conversation budget state. Carried on ``LoopState``, never reset.

    ``seen_ids`` freezes every result that has passed the budget check;
    ``replacements`` maps the subset that was replaced to the exact preview
    string shown to the model.
    """

    seen_ids: set[str] = field(default_factory=set)
    replacements: dict[str, str] = field(default_factory=dict)


def _preview(content: str, tool_call_id: str, output_dir: Path | None) -> str:
    """Spill ``content`` to disk and build the preview string shown to the model."""
    head = content[:PREVIEW_CHARS]
    path = save_overflow_to(content, f"budget-{tool_call_id}", output_dir)
    note = (
        f" Full result ({len(content)} chars) saved to {path} — use the Read tool to view it."
        if path is not None
        else f" (full result was {len(content)} chars; truncated, not saved)."
    )
    return f"{head}\n... [tool result truncated by budget.{note}]"


def apply_tool_result_budget(
    messages: list[dict],
    state: ContentReplacementState,
    output_dir: Path | None = None,
) -> list[dict]:
    """Return a view of ``messages`` with over-budget tool results replaced.

    Mutates ``state`` (records the decisions). Returns the same list object
    untouched when nothing was replaced or re-applied (keeps the no-op cheap).
    """
    changed = False
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") != "tool":
            out.append(msg)
            continue
        tcid = msg.get("tool_call_id")

        # Frozen: re-apply the exact cached preview (byte-identical, cannot fail).
        if tcid in state.replacements:
            out.append({**msg, "content": state.replacements[tcid]})
            changed = True
            continue
        # Seen before and left alone — never replace it later (cache stability).
        if tcid in state.seen_ids:
            out.append(msg)
            continue

        # Fresh result: decide its fate now and freeze it.
        state.seen_ids.add(tcid)
        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= PER_RESULT_CHAR_CAP:
            out.append(msg)
            continue

        preview = _preview(content, str(tcid), output_dir)
        state.replacements[tcid] = preview
        out.append({**msg, "content": preview})
        changed = True

    return out if changed else messages
