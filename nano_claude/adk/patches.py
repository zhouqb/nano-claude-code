"""Targeted runtime patches to the *pinned* google-adk internals.

Two small behaviors of the pinned ADK don't fit the agent contract, and
neither has an extension point. Both patches are tiny, idempotent, and tied
to the exact pin in pyproject.toml — revisit them on any ADK upgrade (the
wire-fidelity tests in tests/test_adk_convert.py are the canary).

1. **Malformed tool-call argument JSON must not kill the turn.** Models
   occasionally stream unparseable argument JSON. ADK raises
   ``json.JSONDecodeError`` from deep inside response aggregation, erroring
   the whole invocation; the old loop answered the call with an error tool
   message the model could recover from. The lenient parser returns a
   sentinel dict instead, and ``before_tool_callback`` converts it into the
   old error message.

2. **Skip trace serialization when nobody is recording.** ADK's
   ``trace_call_llm`` JSON-serializes the full request on every LLM call even
   when the OTel tracer is a no-op — wasted work, and it raises pydantic
   serializer warnings for the raw-string function responses our converter
   deliberately carries (see ``convert.py``). Skip it for non-recording
   spans and silence the (expected) warning otherwise.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import google.adk.flows.llm_flows.base_llm_flow as _base_llm_flow
import google.adk.models.lite_llm as _lite_llm
from opentelemetry import trace as _otel_trace

# Sentinel keys marking arguments that failed JSON parsing. The raw text and
# the parse error ride along so the error message matches the old loop's.
INVALID_JSON_ARGS_KEY = "__nano_invalid_tool_json__"
INVALID_JSON_ERROR_KEY = "__nano_invalid_tool_json_error__"

_applied = False
_original_parse = _lite_llm._parse_tool_call_arguments
_original_trace_call_llm = _base_llm_flow.trace_call_llm


def invalid_json_error(args: dict[str, Any]) -> str | None:
    """The parse-error string if ``args`` is the lenient parser's sentinel."""
    if isinstance(args, dict) and INVALID_JSON_ARGS_KEY in args:
        return str(args.get(INVALID_JSON_ERROR_KEY, "unparseable arguments"))
    return None


def _lenient_parse_tool_call_arguments(arguments: Any) -> Any:
    try:
        return _original_parse(arguments)
    except json.JSONDecodeError as exc:
        return {
            INVALID_JSON_ARGS_KEY: arguments if isinstance(arguments, str) else str(arguments),
            INVALID_JSON_ERROR_KEY: str(exc),
        }


def _quiet_trace_call_llm(*args: Any, **kwargs: Any) -> None:
    if not _otel_trace.get_current_span().is_recording():
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        _original_trace_call_llm(*args, **kwargs)


def apply() -> None:
    """Install the patches (idempotent; called on driver import)."""
    global _applied
    if _applied:
        return
    _applied = True
    _lite_llm._parse_tool_call_arguments = _lenient_parse_tool_call_arguments
    _base_llm_flow.trace_call_llm = _quiet_trace_call_llm
