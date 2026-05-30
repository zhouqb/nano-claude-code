"""Layer 1 — Budget Reduction.

Caps the aggregate size of each tool-result *batch* (the run of consecutive
``role: "tool"`` messages answering one assistant turn's ``tool_calls``). When a
batch exceeds ``BATCH_CHAR_BUDGET``, its largest fresh results are spilled to
disk (largest-first until under budget) and replaced in the view with a preview
pointing at the saved file; ``state.messages`` keeps the full content. Mirrors
Claude Code's ``enforceToolResultBudget`` / ``selectFreshToReplace``.

Decisions are frozen per ``tool_call_id`` to keep the prompt-cache prefix stable
across turns: a replaced result re-emits the same preview (byte-identical), and a
seen-but-unreplaced result is never replaced later. Spill paths are derived
deterministically from the ``tool_call_id`` so the preview is identical across
``--resume``.

``PreviewFormat`` selects the excerpt shape: ``"prefix"`` (head only, default) or
``"head_tail"`` (head + tail, middle elided).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nano_claude.tools.overflow import save_overflow_to

# Aggregate cap on a single tool-result batch, in characters (~5k tokens). Claude
# Code's per-message budget is MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200_000 (~50k
# tokens, GrowthBook-overridable); nano runs ~10x tighter on purpose, so Layer 1
# actually engages in a small session instead of deferring everything to L3–L5.
BATCH_CHAR_BUDGET = 20_000
# Inline preview kept from a spilled result, in characters. Matches Claude Code's
# PREVIEW_SIZE_BYTES = 2000.
PREVIEW_CHARS = 2000

# How a spilled result is excerpted inline. Both shapes are pure functions of the
# content, so the preview stays byte-stable across turns (see module docstring).
#   "prefix"    – keep the first PREVIEW_CHARS (cheapest; the start is usually the
#                 most informative part of tool output).
#   "head_tail" – keep the head AND tail, eliding the middle (when the end matters
#                 too, e.g. a stack trace's final frames or a command's exit line).
PreviewFormat = Literal["prefix", "head_tail"]
DEFAULT_PREVIEW_FORMAT: PreviewFormat = "prefix"

_ELISION = "\n... [middle elided — full result on disk] ...\n"


@dataclass
class ContentReplacementState:
    """Per-conversation budget state. Carried on ``LoopState``, never reset.

    ``seen_ids`` freezes every result that has passed the budget check;
    ``replacements`` maps the subset that was replaced to the exact preview
    string shown to the model.
    """

    seen_ids: set[str] = field(default_factory=set)
    replacements: dict[str, str] = field(default_factory=dict)


def _content_len(msg: dict) -> int:
    content = msg.get("content")
    return len(content) if isinstance(content, str) else 0


def _excerpt(content: str, fmt: PreviewFormat) -> str:
    """The inline slice kept from a spilled result; pure in ``content`` and ``fmt``.

    ``"head_tail"`` splits ``PREVIEW_CHARS`` between the head and tail and elides
    the middle; it falls back to a plain prefix when the content is too short to
    have a distinct middle. ``"prefix"`` keeps the first ``PREVIEW_CHARS``.
    """
    if fmt == "head_tail" and len(content) > PREVIEW_CHARS:
        half = PREVIEW_CHARS // 2
        return f"{content[:half]}{_ELISION}{content[-half:]}"
    return content[:PREVIEW_CHARS]


def _preview(content: str, tool_call_id: str, output_dir: Path | None, fmt: PreviewFormat) -> str:
    """Spill ``content`` to disk and build the preview string shown to the model."""
    excerpt = _excerpt(content, fmt)
    path = save_overflow_to(content, f"budget-{tool_call_id}", output_dir)
    note = (
        f" Full result ({len(content)} chars) saved to {path} — use the Read tool to view it."
        if path is not None
        else f" (full result was {len(content)} chars; truncated, not saved)."
    )
    return f"{excerpt}\n... [tool result truncated by budget.{note}]"


def _select_fresh_to_replace(fresh: list[dict], frozen_size: int) -> set[str]:
    """Largest-first: pick fresh results to spill until the batch fits the budget.

    ``fresh`` are this batch's never-seen results with string content. Returns the
    set of ``tool_call_id``s to replace. If the frozen remainder alone already
    exceeds the budget, we still spill all fresh (best effort) and accept the
    overage — a later layer handles it.
    """
    running = frozen_size + sum(_content_len(m) for m in fresh)
    selected: set[str] = set()
    for m in sorted(fresh, key=_content_len, reverse=True):
        if running <= BATCH_CHAR_BUDGET:
            break
        selected.add(m.get("tool_call_id"))
        running -= _content_len(m)
    return selected


def _process_batch(
    batch: list[dict],
    state: ContentReplacementState,
    output_dir: Path | None,
    fmt: PreviewFormat,
) -> tuple[list[dict], bool]:
    """Apply the aggregate budget to one tool-result batch; update ``state``."""
    frozen_size = 0  # seen-but-unreplaced full content that must stay
    fresh: list[dict] = []  # never-seen, string content → eligible to spill
    for m in batch:
        tcid = m.get("tool_call_id")
        if tcid in state.replacements:
            continue  # already replaced; preview is negligible, ignore for budget
        if tcid in state.seen_ids:
            frozen_size += _content_len(m)
        elif isinstance(m.get("content"), str):
            fresh.append(m)

    # Fast path (mirrors CC's `if (fresh.length === 0) continue`): a batch with no
    # fresh results carries no new content to budget, so skip the eviction search —
    # every id just re-applies its already-frozen fate below. A given batch is thus
    # budgeted exactly once, on the turn it first appears; later turns only re-apply.
    if fresh:
        over_budget = frozen_size + sum(_content_len(m) for m in fresh) > BATCH_CHAR_BUDGET
        selected = _select_fresh_to_replace(fresh, frozen_size) if over_budget else set()
    else:
        selected = set()

    out: list[dict] = []
    changed = False
    for m in batch:
        tcid = m.get("tool_call_id")
        if tcid in state.replacements:
            out.append({**m, "content": state.replacements[tcid]})
            changed = True
            continue
        # First sight of this id (re-applied/seen ids took the branches above):
        # freeze its fate now so it is never re-budgeted on a later turn.
        state.seen_ids.add(tcid)
        if tcid in selected:
            preview = _preview(m["content"], str(tcid), output_dir, fmt)
            state.replacements[tcid] = preview
            out.append({**m, "content": preview})
            changed = True
        else:
            out.append(m)
    return out, changed


def apply_tool_result_budget(
    messages: list[dict],
    state: ContentReplacementState,
    output_dir: Path | None = None,
    *,
    preview_format: PreviewFormat = DEFAULT_PREVIEW_FORMAT,
) -> list[dict]:
    """Return a view of ``messages`` with over-budget tool-result batches replaced.

    ``preview_format`` selects how spilled results are excerpted inline (see
    ``PreviewFormat``); it affects only results spilled *now* — already-frozen
    replacements re-emit their stored string unchanged, so changing the format
    mid-session never rewrites cached history.

    Mutates ``state`` (records the decisions). Returns the same list object
    untouched when nothing was replaced or re-applied (keeps the no-op cheap).
    """
    changed = False
    out: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        if messages[i].get("role") != "tool":
            out.append(messages[i])
            i += 1
            continue
        # Collect the consecutive tool-result batch (one assistant turn's results).
        start = i
        while i < n and messages[i].get("role") == "tool":
            i += 1
        processed, batch_changed = _process_batch(
            messages[start:i], state, output_dir, preview_format
        )
        out.extend(processed)
        changed = changed or batch_changed

    return out if changed else messages
