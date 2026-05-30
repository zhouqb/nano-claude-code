"""Layer 1 — Budget Reduction.

Caps the *aggregate* size of each tool-result batch. A "batch" is the run of
consecutive ``role: "tool"`` messages answering one assistant turn's
``tool_calls`` — nano's equivalent of the single user message that carries N
``tool_result`` blocks in Claude Code. When a batch's combined size exceeds
``BATCH_CHAR_BUDGET``, the *largest* fresh results are spilled to disk and
replaced in the view with a preview pointing the model at the saved file
(largest-first eviction until the batch is back under budget). The canonical
``state.messages`` keeps full content for storage/scrollback.

This mirrors Claude Code's ``enforceToolResultBudget`` / ``selectFreshToReplace``:
budgeting across the batch (not per result) is what catches the case where many
medium results in one round overflow the next prompt even though none of them is
individually large.

The decisive property is **cache-stable decisions** keyed by ``tool_call_id``.
This is *not* "we edit cached history cheaply" — editing any token mid-prompt
invalidates the cache from that point on, and nothing here changes that. The
guarantee is narrower, and comes in two parts:

  1. **The full result never enters the sent prefix.** A result is spilled the
     first time the pipeline sees it — a fresh tail message, produced this very
     turn, past the cache frontier and not yet transmitted. So replacing it
     destroys no cache: the bytes it replaces were never sent. Freezing enforces
     this: a result is either spilled on first sight or added to ``seen_ids`` and
     **never replaced later**. There is no "sent in full on turn N, spilled on
     turn N+3" path — that path is the one that *would* rewrite already-cached
     history, and it cannot happen.

  2. **The preview that does enter the prefix never changes.** Once a preview is
     in the prefix, later turns append after it, so its bytes must be identical
     every turn or the cache breaks at its position. Two things guarantee that:
     the frozen ``replacements[tcid]`` re-emits the exact same string (the budget
     is never re-evaluated, so the decision can't flip), and the spill path is
     derived deterministically from ``tool_call_id`` (no timestamp/uuid), so the
     preview is byte-identical even across ``--resume``, when the decision is
     re-derived from an empty state — no leaked files, no churn.

So nothing cached is destroyed when a result is spilled (part 1), and nothing
already in the prefix is later mutated (part 2). A non-deterministic preview —
one embedding a timestamp or uuid — would violate part 2 and re-invalidate the
cache at its position every single turn; hence the deterministic path.

Two preview shapes are supported (``PreviewFormat``), both pure functions of the
content (so part 2 holds for either): ``"prefix"`` (default) keeps the head;
``"head_tail"`` keeps the head and tail and elides the middle, for output whose
end also matters (a stack trace's final frames, a command's exit summary). The
full content is always spilled to disk for the model to ``Read`` back regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nano_claude.tools.overflow import save_overflow_to

# Aggregate cap on a single tool-result batch, in characters (~5k tokens).
# (Claude Code sources its per-message limit from GrowthBook; nano uses a constant.)
BATCH_CHAR_BUDGET = 20_000
# Total inline budget for a spilled result's preview, in characters.
PREVIEW_CHARS = 500

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

    over_budget = frozen_size + sum(_content_len(m) for m in fresh) > BATCH_CHAR_BUDGET
    selected = _select_fresh_to_replace(fresh, frozen_size) if over_budget else set()

    out: list[dict] = []
    changed = False
    for m in batch:
        tcid = m.get("tool_call_id")
        if tcid in state.replacements:
            out.append({**m, "content": state.replacements[tcid]})
            changed = True
            continue
        state.seen_ids.add(tcid)  # freeze this result's fate
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
