"""Marker-gated live integration test: one real turn through the full stack.

Run with ``pytest -m live`` and a provider API key for the default model.
This is the upgrade canary for google-adk / litellm bumps: it exercises real
streaming, tool-call delta aggregation, and id preservation end-to-end.
"""

from __future__ import annotations

import os

import pytest

from nano_claude.adk.driver import run_turn
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.settings import Settings

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("NANO_CLAUDE_LIVE_TEST")),
    reason="live test: needs DEEPSEEK_API_KEY (or NANO_CLAUDE_LIVE_TEST=1)",
)
async def test_one_real_turn_with_tool_call(tmp_path):
    (tmp_path / "marker.txt").write_text("nano-claude live test marker\n")

    state = LoopState(
        messages=[
            {"role": "system", "content": "You are a terse test agent."},
            {
                "role": "user",
                "content": (
                    f"Use the Read tool to read {tmp_path / 'marker.txt'} "
                    "and then reply with just the word it contains after 'nano-claude'."
                ),
            },
        ]
    )
    config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS, max_turns=5)

    result = await run_turn(state, config, settings=Settings(path=tmp_path / "s.json"))

    assert result.reason is StopReason.COMPLETED
    assert "live" in result.final_text.lower()
    # The transcript holds a real provider tool-call id, matched by its result.
    assistant = next(m for m in state.messages if m.get("tool_calls"))
    (tc,) = assistant["tool_calls"]
    tool_msg = next(m for m in state.messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == tc["id"]
