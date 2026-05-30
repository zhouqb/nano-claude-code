"""The context-management pipeline — the orchestrator for compaction.

Run once per turn, before the model call. The layers run in a fixed order,
cheapest / most-granular / most-reversible first, lossiest last, so an earlier
layer that frees enough makes the later ones no-op and granular context
survives as long as possible:

    1. Budget Reduction   per-result size cap → spill big results to disk
    2. Snip               prune zombie/stale messages (structural, no model)
    3. Microcompact       clear OLD tool-result *content*, keep recent N
    4. Context Collapse   archive read/search spans → projected view
    5. Auto-Compact       summarize EVERYTHING → replace history

**View vs. canonical store.** Layers 1/3/4 produce a *derived view* sent to the
model while ``state.messages`` stays canonical for storage and scrollback.
Layer 2 may prune the canonical store; Layer 5 replaces it. This module returns
the view the loop should send.

This is the *foundation*: it establishes the seam (``run_context_management``
returns the view; the loop sends it instead of ``state.messages``) and wires
only Layer 5 + the warning/blocking gates. Layers 1–4 land in later PRs, each
slotting into the marked points below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nano_claude.compaction.auto_compact import (
    circuit_broken,
    should_auto_compact,
    should_block,
    should_warn,
)
from nano_claude.compaction.compactor import compact_conversation

if TYPE_CHECKING:
    from nano_claude.agent.loop import LoopCallbacks
    from nano_claude.agent.types import AgentConfig, LoopState


@dataclass
class ContextView:
    """Outcome of one pipeline pass."""

    messages: list[dict]  # the message view to send to the model this turn
    blocked: bool = False  # True → context is full and auto-compact can't help


async def run_context_management(
    state: LoopState,
    config: AgentConfig,
    callbacks: LoopCallbacks,
) -> ContextView:
    """Run the compaction layers and return the message view for this turn.

    The canonical history lives in ``state.messages``. View-only layers (1/3/4,
    landing later) will build on a copy; in this foundation the only active
    transform is Layer 5 (auto-compaction), which replaces ``state.messages`` in
    place — so the returned view is ``state.messages`` itself.
    """
    # NOTE: future view-only layers start from a copy and transform it:
    #     view = list(state.messages)
    #     view = apply_tool_result_budget(view, state)   # Layer 1
    #     view = snip_messages(view).messages            # Layer 2 (may prune canonical too)
    #     view = microcompact(view)                      # Layer 3
    #     view = apply_collapses(view, state, config)    # Layer 4

    # --- Layer 5: Auto-Compact -------------------------------------------
    # Last resort: summarize the whole conversation and replace it. When it
    # runs it mutates state.messages, so the view tracks the canonical store.
    if should_auto_compact(state, config):
        compacted = await compact_conversation(state, config)
        if compacted:
            if callbacks.on_compact:
                callbacks.on_compact()
        elif circuit_broken(state):
            # Repeated failures: stop trying for the rest of the session so we
            # don't hammer the API with doomed compaction attempts.
            config.auto_compact = False
            if callbacks.on_compact_disabled:
                callbacks.on_compact_disabled()
    elif should_warn(state, config) and callbacks.on_context_warning:
        callbacks.on_context_warning()

    # --- Blocking gate ---------------------------------------------------
    # Only meaningful when auto-compact is off (disabled by config or tripped
    # circuit breaker): refuse the call so headroom is reserved for a manual
    # /compact. A successful compaction above reset the token signal, so this
    # is naturally skipped right after compacting.
    blocked = not config.auto_compact and should_block(state, config)

    return ContextView(messages=state.messages, blocked=blocked)
