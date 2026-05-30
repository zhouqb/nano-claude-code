"""Tests for the compaction pipeline foundation.

Covers the orchestrator (run_context_management) wiring of Layer 5 + the
warning/blocking gates, and the loop's view/canonical seam + BLOCKED path.
"""

from __future__ import annotations

import litellm

from nano_claude.agent.loop import LoopCallbacks, query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.compaction.pipeline import run_context_management
from tests.conftest import make_acompletion, text_chunk, usage_chunk


def _state(tokens: int, **kw) -> LoopState:
    return LoopState(messages=[{"role": "user", "content": "x"}], last_input_tokens=tokens, **kw)


def _flag_callbacks() -> tuple[LoopCallbacks, dict]:
    """Callbacks that record which pipeline events fired."""
    fired = {"compact": False, "disabled": False, "warn": False}
    return (
        LoopCallbacks(
            on_compact=lambda: fired.__setitem__("compact", True),
            on_compact_disabled=lambda: fired.__setitem__("disabled", True),
            on_context_warning=lambda: fired.__setitem__("warn", True),
        ),
        fired,
    )


# --- below thresholds: pure pass-through ------------------------------------


async def test_passthrough_returns_canonical_view():
    config = AgentConfig(context_window=200_000)
    state = _state(100_000)
    cbs, fired = _flag_callbacks()

    view = await run_context_management(state, config, cbs)

    # The view is a derived copy with identical content (layers must not mutate
    # the canonical store), and nothing fired below threshold.
    assert view.messages == state.messages
    assert view.messages is not state.messages
    assert view.blocked is False
    assert fired == {"compact": False, "disabled": False, "warn": False}


# --- warning gate -----------------------------------------------------------


async def test_warning_fires_between_warn_and_compact_thresholds():
    # window 200k → warn at 180k, auto-compact at 187k. 183k warns but doesn't compact.
    config = AgentConfig(context_window=200_000)
    state = _state(183_000)
    cbs, fired = _flag_callbacks()

    await run_context_management(state, config, cbs)

    assert fired["warn"] is True
    assert fired["compact"] is False


# --- Layer 5 trigger --------------------------------------------------------


async def test_auto_compact_fires_and_replaces_history(monkeypatch):
    config = AgentConfig(context_window=200_000)
    state = _state(190_000)  # above auto-compact threshold (187k)
    state.messages = [
        {"role": "system", "content": "sys"},
        *({"role": "user", "content": f"m{i}"} for i in range(10)),
    ]
    cbs, fired = _flag_callbacks()

    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_acompletion([text_chunk("STRUCTURED SUMMARY"), usage_chunk(5, 5)]),
    )

    view = await run_context_management(state, config, cbs)

    assert fired["compact"] is True
    # warn must not also fire when we compacted (the elif guard).
    assert fired["warn"] is False
    # History was replaced; the summary is present and the signal was reset.
    assert any("STRUCTURED SUMMARY" in str(m.get("content")) for m in state.messages)
    assert state.last_input_tokens == 0
    # View reflects the freshly-compacted canonical store.
    assert view.messages == state.messages


# --- circuit breaker --------------------------------------------------------


async def test_circuit_breaker_disables_autocompact(monkeypatch):
    config = AgentConfig(context_window=200_000)
    # One more failure trips the breaker (MAX = 3).
    state = _state(190_000, consecutive_compact_failures=2)
    cbs, fired = _flag_callbacks()

    async def _boom(*a, **k):
        raise RuntimeError("summary API down")

    monkeypatch.setattr(litellm, "acompletion", _boom)

    await run_context_management(state, config, cbs)

    assert state.consecutive_compact_failures == 3
    assert config.auto_compact is False
    assert fired["disabled"] is True
    assert fired["compact"] is False


# --- blocking gate ----------------------------------------------------------


async def test_blocking_only_when_autocompact_off():
    config = AgentConfig(context_window=200_000)  # block at 197k
    state = _state(198_000)

    # Auto-compact on: never block (Layer 5 owns headroom).
    assert (await run_context_management(state, config, LoopCallbacks())).blocked is False

    # Auto-compact off: block to reserve room for manual /compact.
    config.auto_compact = False
    assert (await run_context_management(state, config, LoopCallbacks())).blocked is True


async def test_loop_returns_blocked_without_calling_model(monkeypatch):
    config = AgentConfig(context_window=200_000, auto_compact=False)
    state = _state(198_000)

    async def _should_not_be_called(*a, **k):
        raise AssertionError("model must not be called when blocked")

    monkeypatch.setattr(litellm, "acompletion", _should_not_be_called)

    result = await query_loop(state, config)

    assert result.reason is StopReason.BLOCKED
    assert "compact" in result.final_text.lower()
