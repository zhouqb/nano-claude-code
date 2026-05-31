"""Tests for the ReplUI rendering state machine (headless console)."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from nano_claude.tools.base import ToolResult
from nano_claude.ui import ReplUI, _summarize_args


def _ui() -> tuple[ReplUI, StringIO]:
    buf = StringIO()
    return ReplUI(Console(file=buf, force_terminal=False, width=100)), buf


# --- arg summary ------------------------------------------------------------


def test_summarize_args_prefers_known_keys():
    assert _summarize_args({"command": "ls -la"}) == "ls -la"
    assert _summarize_args({"file_path": "/x/y.py"}) == "/x/y.py"
    assert _summarize_args({"subagent_type": "explorer"}) == "explorer"


def test_summarize_args_falls_back_to_kwargs():
    assert "foo" in _summarize_args({"foo": "bar"})


# --- spinner state machine --------------------------------------------------


def test_request_start_starts_spinner_assistant_stops_it():
    ui, buf = _ui()
    ui.on_request_start()
    assert ui._status is not None
    ui.on_assistant_start()
    assert ui._status is None  # spinner stopped before any text prints
    assert "assistant" in buf.getvalue()


def test_tool_start_stops_spinner_and_prints_header():
    ui, buf = _ui()
    ui.on_request_start()
    ui.on_tool_start("Bash", {"command": "ls"})
    assert ui._status is None
    out = buf.getvalue()
    assert "Bash" in out
    assert "ls" in out


def test_double_request_start_is_idempotent():
    ui, _ = _ui()
    ui.on_request_start()
    first = ui._status
    ui.on_request_start()
    assert ui._status is first  # no second spinner


def test_finish_turn_is_safe_with_no_spinner():
    ui, _ = _ui()
    ui.finish_turn()  # must not raise
    assert ui._status is None


def test_streaming_newline_closed_on_tool_start():
    ui, buf = _ui()
    ui.on_assistant_start()
    ui.on_text("partial")
    ui.on_tool_start("Read", {"file_path": "/a"})
    # The streamed line is closed (newline) before the tool header.
    assert "partial" in buf.getvalue()
    assert not ui._streaming


# --- tool result preview ----------------------------------------------------


def test_tool_end_shows_more_lines_hint():
    ui, buf = _ui()
    ui.on_tool_end("Read", ToolResult(output="line1\nline2\nline3"))
    out = buf.getvalue()
    assert "line1" in out
    assert "+2 more lines" in out


def test_callbacks_wires_request_start():
    ui, _ = _ui()
    cb = ui.callbacks()
    assert cb.on_request_start is not None
    assert cb.on_tool_end is not None
