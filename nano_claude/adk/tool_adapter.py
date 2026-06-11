"""Wrap a nano ``Tool`` as an ADK ``BaseTool``.

One adapter instance is built per tool *per turn*, with the turn's
:class:`~nano_claude.tools.base.ToolContext` bound in — that context carries
live, non-serializable state (cancel event, prompter, file-read snapshots), so
it deliberately does not travel through ADK's ``tool_context.state``.

Responsibilities ported verbatim from the old loop's ``_run_call``:
the ``tool <name>`` OTel span, the cancel-event short-circuit, and the
never-let-a-tool-crash-the-loop containment. Argument validation also happens
here (the old ``_resolve_call`` validated once; under ADK the permission
callback and the tool both receive raw dicts, and re-validating is cheap and
side-effect-free). Permission gating and hooks live in the ADK callbacks, not
here.

``run_async`` returns the result *string*; ADK wraps non-dict returns as
``{"result": <str>}`` on the recorded event, which ``convert.event_to_messages``
unwraps back to the raw string.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from google.adk.tools import BaseTool
from google.adk.tools import ToolContext as AdkToolContext
from google.genai import types
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode
from pydantic import ValidationError

from nano_claude.telemetry import log, set_content_attribute, tracer
from nano_claude.tools.base import Tool, ToolContext, ToolResult


class NanoToolAdapter(BaseTool):
    """Adapt a :class:`nano_claude.tools.base.Tool` to ADK's tool protocol."""

    def __init__(
        self,
        tool: Tool,
        context: ToolContext,
        *,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, ToolResult], None] | None = None,
    ) -> None:
        super().__init__(name=tool.name, description=tool.description)
        self._tool = tool
        self._context = context
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    @property
    def tool(self) -> Tool:
        """The wrapped nano tool (the permission callback dispatches on it)."""
        return self._tool

    def _get_declaration(self) -> types.FunctionDeclaration:
        # ``to_api_schema`` is the single source of truth for the advertised
        # schema (MCPTool overrides it to pass the server's raw JSON Schema
        # through verbatim). ``parameters_json_schema`` reaches the OpenAI
        # ``tools`` payload unmodified in the pinned ADK's LiteLlm converter.
        fn = self._tool.to_api_schema()["function"]
        return types.FunctionDeclaration(
            name=fn["name"],
            description=fn["description"],
            parameters_json_schema=fn["parameters"],
        )

    async def run_async(self, *, args: dict[str, Any], tool_context: AdkToolContext) -> Any:
        if self._context.cancel_event.is_set():
            return "[Interrupted]"

        if self._tool.reads_raw_args:
            args_model: Any = args
        else:
            try:
                args_model = self._tool.input_schema.model_validate(args)
            except ValidationError as exc:
                return f"Error: invalid arguments for {self.name}: {exc}"

        with tracer.start_as_current_span(f"tool {self.name}", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("nano_claude.tool.name", self.name)
            set_content_attribute(span, "nano_claude.tool.arguments", args or {})
            if self._on_tool_start:
                self._on_tool_start(self.name, args or {})
            try:
                result = await self._tool.call(args_model, self._context)
            except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                result = ToolResult.fail(f"Tool raised an exception: {exc}")
            span.set_attribute("nano_claude.tool.is_error", result.is_error)
            set_content_attribute(span, "nano_claude.tool.output", result.output)
            if result.is_error:
                span.set_attribute("nano_claude.tool.error", result.error or "tool error")
                span.set_status(Status(StatusCode.ERROR, result.error or "tool error"))
            if self._on_tool_end:
                self._on_tool_end(self.name, result)
            log.info("tool %s finished (error=%s)", self.name, result.is_error)
            return result.output
