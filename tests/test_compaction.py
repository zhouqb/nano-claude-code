"""Unit tests for the compaction core: thresholds, counting, compactor."""

from __future__ import annotations

import litellm

from nano_claude.agent.types import AgentConfig, LoopState
from nano_claude.compaction.auto_compact import (
    circuit_broken,
    should_auto_compact,
    should_block,
    should_warn,
)
from nano_claude.compaction.compactor import (
    RECENT_MESSAGES_KEPT,
    SUMMARY_PROMPT,
    build_compact_user_message,
    compact_conversation,
    format_compact_summary,
)
from nano_claude.compaction.thresholds import (
    MAX_CONSECUTIVE_COMPACT_FAILURES,
    auto_compact_threshold,
    block_threshold,
    warn_threshold,
)
from nano_claude.compaction.token_counter import (
    current_context_tokens,
    estimate_message_tokens,
)
from nano_claude.session.storage import SessionStorage, session_file
from tests.conftest import make_acompletion, text_chunk, usage_chunk

# --- thresholds -------------------------------------------------------------


def test_thresholds_scale_with_window():
    assert auto_compact_threshold(200_000) == 187_000
    assert warn_threshold(200_000) == 180_000
    assert block_threshold(200_000) == 197_000
    # Works for a small window too.
    assert auto_compact_threshold(64_000) == 51_000


# --- token counting ---------------------------------------------------------


def test_estimate_message_tokens_nonzero():
    msgs = [{"role": "user", "content": "hello world " * 100}]
    assert estimate_message_tokens(msgs) > 0


def test_current_context_prefers_reported_tokens():
    state = LoopState(messages=[{"role": "user", "content": "x"}], last_input_tokens=1234)
    assert current_context_tokens(state) == 1234


def test_current_context_falls_back_to_estimate():
    state = LoopState(messages=[{"role": "user", "content": "x" * 400}])
    assert current_context_tokens(state) == estimate_message_tokens(state.messages)


# --- threshold predicates ---------------------------------------------------


def _state(tokens: int, **kw) -> LoopState:
    return LoopState(messages=[{"role": "user", "content": "x"}], last_input_tokens=tokens, **kw)


def test_should_auto_compact_respects_threshold():
    config = AgentConfig(context_window=100_000)  # threshold = 87_000
    assert should_auto_compact(_state(90_000), config)
    assert not should_auto_compact(_state(50_000), config)


def test_should_auto_compact_disabled_when_off():
    config = AgentConfig(context_window=100_000, auto_compact=False)
    assert not should_auto_compact(_state(99_000), config)


def test_should_auto_compact_circuit_broken():
    config = AgentConfig(context_window=100_000)
    state = _state(99_000, consecutive_compact_failures=MAX_CONSECUTIVE_COMPACT_FAILURES)
    assert circuit_broken(state)
    assert not should_auto_compact(state, config)


def test_warn_and_block_gates():
    config = AgentConfig(context_window=100_000)  # warn=80k, block=97k
    assert should_warn(_state(85_000), config)
    assert not should_block(_state(85_000), config)
    assert should_block(_state(98_000), config)


# --- compactor --------------------------------------------------------------


def _long_conversation() -> list[dict]:
    msgs = [{"role": "system", "content": "system prompt"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


async def test_compact_replaces_history(tmp_path, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_acompletion([text_chunk("SUMMARY TEXT"), usage_chunk(5, 5)]),
    )
    state = LoopState(messages=_long_conversation(), last_input_tokens=99_000, turn_count=10)
    config = AgentConfig(context_window=100_000)

    ok = await compact_conversation(state, config)

    assert ok
    # Invariant: messages[0] is system, messages[1] is the summary.
    assert state.messages[0]["role"] == "system"
    assert state.messages[1]["role"] == "user"
    assert "SUMMARY TEXT" in state.messages[1]["content"]
    # History shrank and the token counter was reset.
    assert len(state.messages) <= 2 + RECENT_MESSAGES_KEPT
    assert state.last_input_tokens == 0
    assert state.consecutive_compact_failures == 0
    # Most recent turn is preserved verbatim.
    assert state.messages[-1] == {"role": "assistant", "content": "answer 9"}


async def test_compact_persists_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "acompletion",
        make_acompletion([text_chunk("S"), usage_chunk(1, 1)]),
    )
    path = session_file(str(tmp_path), "sid", root=tmp_path / "root")
    storage = SessionStorage(path, "sid")
    state = LoopState(messages=_long_conversation(), last_input_tokens=99_000, storage=storage)

    await compact_conversation(state, AgentConfig(context_window=100_000))
    await storage.flush()

    text = path.read_text()
    assert "compact_boundary" in text


async def test_compact_failure_increments_counter(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("model down")

    monkeypatch.setattr(litellm, "acompletion", _boom)
    state = LoopState(messages=_long_conversation(), last_input_tokens=99_000)

    ok = await compact_conversation(state, AgentConfig(context_window=100_000))

    assert not ok
    assert state.consecutive_compact_failures == 1
    # Messages untouched on failure.
    assert state.messages[0]["content"] == "system prompt"


async def test_compact_empty_summary_is_failure(monkeypatch):
    monkeypatch.setattr(litellm, "acompletion", make_acompletion([usage_chunk(1, 1)]))
    state = LoopState(messages=_long_conversation(), last_input_tokens=99_000)

    ok = await compact_conversation(state, AgentConfig(context_window=100_000))

    assert not ok
    assert state.consecutive_compact_failures == 1


async def test_compact_tail_drops_orphan_tool_message(monkeypatch):
    monkeypatch.setattr(
        litellm, "acompletion", make_acompletion([text_chunk("S"), usage_chunk(1, 1)])
    )
    # Build a history whose -RECENT tail would start mid tool-exchange.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(5):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    msgs.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        }
    )
    msgs.append({"role": "tool", "tool_call_id": "c1", "content": "result"})
    state = LoopState(messages=msgs, last_input_tokens=99_000)

    ok = await compact_conversation(state, AgentConfig(context_window=100_000))
    assert ok
    # No leading orphan tool message after the summary.
    assert state.messages[2]["role"] != "tool"


# --- prompt + summary formatting (verbatim from Claude Code) ----------------


def test_summary_prompt_is_claude_code_verbatim():
    # Spot-check the load-bearing fragments of getCompactPrompt().
    assert SUMMARY_PROMPT.startswith("CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.")
    assert "1. Primary Request and Intent:" in SUMMARY_PROMPT
    assert "6. All user messages:" in SUMMARY_PROMPT
    assert "9. Optional Next Step:" in SUMMARY_PROMPT
    assert "<analysis>" in SUMMARY_PROMPT and "<summary>" in SUMMARY_PROMPT
    assert SUMMARY_PROMPT.rstrip().endswith(
        "Tool calls will be rejected and you will fail the task."
    )


def test_format_strips_analysis_and_unwraps_summary():
    raw = "<analysis>\nscratchpad thoughts\n</analysis>\n<summary>\n1. Intent: do X\n</summary>"
    out = format_compact_summary(raw)
    assert "scratchpad thoughts" not in out
    assert out.startswith("Summary:")
    assert "1. Intent: do X" in out


def test_format_passes_through_plain_text():
    # Output without the XML wrappers is returned trimmed, unchanged.
    assert format_compact_summary("  just a plain summary  ") == "just a plain summary"


def test_compact_user_message_matches_claude_code_wrapper():
    msg = build_compact_user_message(
        "Summary:\n1. Intent: X",
        transcript_path="/tmp/sid.jsonl",
        recent_messages_preserved=True,
        suppress_follow_up=True,
    )
    assert msg.startswith(
        "This session is being continued from a previous conversation that ran out of context."
    )
    assert "Summary:\n1. Intent: X" in msg
    assert "read the full transcript at: /tmp/sid.jsonl" in msg
    assert "Recent messages are preserved verbatim." in msg
    assert "Continue the conversation from where it left off" in msg


def test_compact_user_message_minimal():
    # No transcript / no recent / no suppress → just the base wrapper + summary.
    msg = build_compact_user_message("S")
    assert msg == (
        "This session is being continued from a previous conversation that ran out of "
        "context. The summary below covers the earlier portion of the conversation.\n\nS"
    )
