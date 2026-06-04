"""Telemetry wiring tests.

These exercise the opt-in OTel setup with in-memory exporters (so nothing
touches the network), plus the tool-call span emitted by the loop. The global
TracerProvider can only be installed once per process, so a single module-scoped
fixture configures it and the in-memory exporter is cleared between cases.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nano_claude import telemetry
from nano_claude.agent.loop import LoopCallbacks, _CallPlan, _run_call
from nano_claude.permissions.modes import PermissionMode
from nano_claude.tools.base import ToolContext, ToolResult


class _FakeTool:
    def __init__(self, name: str = "fake", result: ToolResult | None = None) -> None:
        self.name = name
        self._result = result or ToolResult(output="ok")

    async def call(self, args, context) -> ToolResult:  # noqa: ANN001
        return self._result


def _context() -> ToolContext:
    return ToolContext(
        cwd=".",
        cancel_event=asyncio.Event(),
        permission_mode=PermissionMode.DEFAULT,
    )


@pytest.fixture(scope="module")
def exporters():
    spans = InMemorySpanExporter()
    logs = InMemoryLogRecordExporter()
    telemetry.configure(span_exporter=spans, log_exporter=logs)
    yield spans, logs
    telemetry.reset_for_testing()


@pytest.fixture(autouse=True)
def _clear(exporters):
    spans, logs = exporters
    spans.clear()
    logs.clear()


def test_init_disabled_without_env(monkeypatch):
    """init_telemetry is a no-op (returns False) when the env flag is unset."""
    monkeypatch.delenv("NANO_CLAUDE_TELEMETRY", raising=False)
    telemetry._initialized = False
    assert telemetry.init_telemetry() is False


async def test_tool_call_emits_span(exporters):
    spans, _ = exporters
    plan = _CallPlan(tool=_FakeTool(), args_model=None, args_dict={"path": "x"})

    out = await _run_call(plan, _context(), LoopCallbacks(), session_id="sess")

    assert out == "ok"
    tool_spans = [s for s in spans.get_finished_spans() if s.name == "tool fake"]
    assert len(tool_spans) == 1
    attrs = tool_spans[0].attributes
    assert attrs["nano_claude.tool.name"] == "fake"
    assert attrs["nano_claude.tool.is_error"] is False
    # Args + output are captured by default.
    assert attrs["nano_claude.tool.arguments"] == '{"path": "x"}'
    assert attrs["nano_claude.tool.output"] == "ok"
    assert "nano_claude.tool.error" not in attrs


async def test_failed_tool_marks_span_error(exporters):
    from opentelemetry.trace.status import StatusCode

    spans, _ = exporters
    plan = _CallPlan(
        tool=_FakeTool(result=ToolResult.fail("boom")),
        args_model=None,
        args_dict={},
    )

    await _run_call(plan, _context(), LoopCallbacks(), session_id="sess")

    tool_spans = [s for s in spans.get_finished_spans() if s.name == "tool fake"]
    assert tool_spans[0].status.status_code is StatusCode.ERROR
    assert tool_spans[0].attributes["nano_claude.tool.is_error"] is True
    assert tool_spans[0].attributes["nano_claude.tool.error"] == "boom"


async def test_content_capture_can_be_disabled(exporters, monkeypatch):
    monkeypatch.setenv("NANO_CLAUDE_TELEMETRY_CAPTURE_CONTENT", "0")
    spans, _ = exporters
    plan = _CallPlan(tool=_FakeTool(), args_model=None, args_dict={"path": "x"})

    await _run_call(plan, _context(), LoopCallbacks(), session_id="sess")

    attrs = [s for s in spans.get_finished_spans() if s.name == "tool fake"][0].attributes
    assert "nano_claude.tool.arguments" not in attrs
    assert "nano_claude.tool.output" not in attrs


def test_set_content_attribute_truncates(monkeypatch):
    monkeypatch.setenv("NANO_CLAUDE_TELEMETRY_MAX_CONTENT_LEN", "10")

    class _Span:
        def __init__(self) -> None:
            self.attrs: dict = {}

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key, value) -> None:  # noqa: ANN001
            self.attrs[key] = value

    span = _Span()
    telemetry.set_content_attribute(span, "k", "x" * 50)
    assert span.attrs["k"].startswith("xxxxxxxxxx…[truncated 40 chars]")


def test_set_content_attribute_structured_stays_valid_json(monkeypatch):
    """A truncated structured value must remain parseable JSON (per-leaf cap)."""
    import json

    monkeypatch.setenv("NANO_CLAUDE_TELEMETRY_MAX_CONTENT_LEN", "10")

    class _Span:
        def __init__(self) -> None:
            self.attrs: dict = {}

        def is_recording(self) -> bool:
            return True

        def set_attribute(self, key, value) -> None:  # noqa: ANN001
            self.attrs[key] = value

    span = _Span()
    messages = [
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "u" * 100},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    ]
    telemetry.set_content_attribute(span, "gen_ai.messages", messages)

    parsed = json.loads(span.attrs["gen_ai.messages"])  # must not raise
    assert [m["role"] for m in parsed] == ["system", "user", "assistant"]
    assert parsed[0]["content"].startswith("ssssssssss…[truncated 90 chars]")
    assert parsed[2]["content"] is None  # non-string leaves pass through


async def test_tool_call_emits_log(exporters):
    _, logs = exporters
    plan = _CallPlan(tool=_FakeTool(), args_model=None, args_dict={})

    await _run_call(plan, _context(), LoopCallbacks(), session_id="sess")

    records = logs.get_finished_logs()
    messages = [r.log_record.body for r in records]
    assert any("tool fake finished" in str(m) for m in messages)
    # The logger routes to OTel only, not the root/stderr handler chain.
    assert logging.getLogger("nano_claude").propagate is False


class _FakeReadableLogRecord:
    """Stand-in for the SDK's ReadableLogRecord (just needs ``to_json``)."""

    def __init__(self, body: str) -> None:
        self._body = body

    def to_json(self, indent=None) -> str:  # noqa: ANN001 - mirrors SDK record
        return f'{{"body": "{self._body}"}}'


def test_session_file_exporter_swaps_files(tmp_path):
    """Each session's records land in its own file; switching closes the old."""
    exporter = telemetry._build_session_file_exporter()
    first = tmp_path / "20260101-aaaa.log.jsonl"
    second = tmp_path / "20260102-bbbb.log.jsonl"

    exporter.set_session_file(first)
    exporter.export([_FakeReadableLogRecord("first")])
    exporter.set_session_file(second)
    exporter.export([_FakeReadableLogRecord("second")])
    exporter.shutdown()

    assert "first" in first.read_text()
    assert "second" in second.read_text()
    # The second session's records must not bleed into the first session's file.
    assert "second" not in first.read_text()


def test_set_session_log_file_is_noop_when_disabled():
    """No file exporter (telemetry off / OTLP logs) → set_session_log_file no-ops."""
    telemetry._session_log_exporter = None
    telemetry.set_session_log_file("/tmp/should-not-be-created.jsonl")  # must not raise


def test_trace_mode_resolution(monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_TELEMETRY_TRACES", raising=False)
    monkeypatch.delenv("NANO_CLAUDE_TELEMETRY_CONSOLE", raising=False)
    assert telemetry._trace_mode() == "otlp"

    monkeypatch.setenv("NANO_CLAUDE_TELEMETRY_TRACES", "off")
    assert telemetry._trace_mode() == "off"

    monkeypatch.setenv("NANO_CLAUDE_TELEMETRY_TRACES", "console")
    assert telemetry._trace_mode() == "console"

    # Legacy alias still selects console when TRACES is unset.
    monkeypatch.delenv("NANO_CLAUDE_TELEMETRY_TRACES", raising=False)
    monkeypatch.setenv("NANO_CLAUDE_TELEMETRY_CONSOLE", "1")
    assert telemetry._trace_mode() == "console"
