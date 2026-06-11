"""ADK callback factories: where the old loop's per-request logic now lives.

The driver builds one set of callbacks per ``run_turn`` (closures over the
turn's ``LoopState``/``AgentConfig``/``TurnGates``), and attaches them to the
``LlmAgent``:

- ``before_model_callback`` is the old pre-request block: memory-prefetch
  drain, todo reminder, the compaction pipeline (whose view *replaces*
  ``llm_request.contents`` wholesale — ADK's own session-derived contents are
  discarded), and the cumulative max-turns / context-blocked gates. A gate
  trips by returning an empty ``LlmResponse``: ADK skips the LLM call, emits
  no event, and ends the invocation; the driver reads the reason off
  :class:`TurnGates`.
- ``before_tool_callback`` is the old ``_resolve_call``: allowed-tools
  restriction, Pydantic validation, the permission manager (interactive
  prompts included — ADK runs tool calls in parallel, and the manager's
  module-level prompt lock keeps prompts serialized), then PreToolUse hooks.
  Returning a ``{"result": ...}`` dict skips execution, exactly like the old
  fixed-content path.
- ``after_tool_callback`` appends PostToolUse hook context to the result.
- ``on_tool_error_callback`` maps an unadvertised/unknown tool name to the
  old error strings instead of ADK's raised ``ValueError``.
- ``after_model_callback`` accounts token usage + turn count from the final
  (non-partial) response of each LLM call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google.adk.models.llm_response import LlmResponse
from opentelemetry import trace as otel_trace
from pydantic import ValidationError

from nano_claude.adk.convert import view_to_contents
from nano_claude.adk.patches import invalid_json_error
from nano_claude.adk.tool_adapter import NanoToolAdapter
from nano_claude.agent.reminders import maybe_todo_reminder
from nano_claude.agent.types import AgentConfig, LoopCallbacks, LoopState, StopReason
from nano_claude.compaction.pipeline import run_context_management
from nano_claude.compaction.token_counter import record_input_tokens_from_usage
from nano_claude.extensibility.hooks import HookEvent, execute_hooks
from nano_claude.permissions.manager import Prompter, has_permission_to_use_tool
from nano_claude.permissions.settings import Settings
from nano_claude.telemetry import log, set_content_attribute
from nano_claude.tools.base import ToolContext
from nano_claude.tools.registry import get_tool

BLOCKED_NOTICE = "Context is full. Run /compact to continue."


@dataclass
class TurnGates:
    """Per-``run_turn`` state shared between the callbacks and the driver."""

    # Stop reason set by a tripped gate; the driver maps it to the LoopResult.
    reason: StopReason | None = None
    notice: str = ""
    # True between a request being issued and its final response — lets
    # after_model count exactly one turn per LLM call even if the stream
    # aggregates into multiple non-partial responses.
    expecting_response: bool = False
    # Tracks on_assistant_start (first text token of each response).
    assistant_started: bool = False


def make_before_model_callback(
    state: LoopState,
    config: AgentConfig,
    callbacks: LoopCallbacks,
    gates: TurnGates,
    *,
    allowed_tools: list[str] | None,
    memory_prefetch: Any,
    record: Any,
):
    async def before_model_callback(*, callback_context, llm_request):
        if state.cancel_event.is_set():
            gates.reason = StopReason.ABORTED
            return LlmResponse()
        if state.turn_count >= config.max_turns:
            gates.reason = StopReason.MAX_TURNS
            return LlmResponse()

        # Memory recall: consume the prefetch only if it has already settled, so
        # a slow side-query never delays the turn. Surfaced files become part of
        # this request's view; if it never settles before the turn ends it's
        # simply dropped (and cancelled by the REPL).
        if memory_prefetch is not None:
            for msg in memory_prefetch.drain_if_ready():
                record(msg)

        # Nudge the model back to the todo list when it's gone stale, so a
        # long task doesn't silently drift off-plan. Recorded like any other
        # message so it persists and resets the reminder turn counter.
        reminder = maybe_todo_reminder(state, allowed_tools)
        if reminder is not None:
            record(reminder)

        # Run the compaction pipeline and send its derived view to the model.
        # state.messages stays canonical (storage + scrollback); the view is
        # what the model sees this request. ADK's session-derived contents are
        # rebuilt from it wholesale.
        view = await run_context_management(state, config, callbacks)
        if view.blocked:
            gates.reason = StopReason.BLOCKED
            gates.notice = BLOCKED_NOTICE
            return LlmResponse()

        instruction, contents = view_to_contents(view.messages)
        llm_request.contents = contents
        llm_request.config.system_instruction = instruction

        # This callback runs inside ADK's per-request LLM span; rename it to
        # the historical ``chat <model>`` and attach the GenAI request
        # attributes there (ADK's own bulk request capture is disabled in
        # patches.py — these attributes are the span's content contract).
        span = otel_trace.get_current_span()
        if span.is_recording():
            span.update_name(f"chat {config.model}")
            span.set_attribute("gen_ai.operation.name", "chat")
            span.set_attribute("gen_ai.request.model", config.model)
            if config.reasoning_effort is not None:
                span.set_attribute("gen_ai.request.reasoning_effort", config.reasoning_effort)
            # ``gen_ai.messages`` holds the request input only; the assistant
            # reply lives in ``gen_ai.response`` (set by after_model). Keeping
            # them as two attributes — input first, response after — makes the
            # exchange easier to read in the span, and a failed/aborted request
            # still shows exactly what was sent.
            set_content_attribute(span, "gen_ai.messages", view.messages)

        gates.expecting_response = True
        gates.assistant_started = False
        if callbacks.on_request_start:
            callbacks.on_request_start()
        return None

    return before_model_callback


def make_after_model_callback(state: LoopState, gates: TurnGates):
    async def after_model_callback(*, callback_context, llm_response):
        if llm_response.partial:
            return None
        usage = llm_response.usage_metadata
        if usage is not None:
            state.token_usage.input_tokens += usage.prompt_token_count or 0
            state.token_usage.output_tokens += usage.candidates_token_count or 0
            state.token_usage.cache_read_tokens += usage.cached_content_token_count or 0
            # Record the API-reported input-token count — the current-context
            # signal the pipeline's thresholds key off next request.
            record_input_tokens_from_usage(state, usage)
        if gates.expecting_response:
            state.turn_count += 1
            gates.expecting_response = False

        # ADK rebinds this callback to the request's LLM span: annotate the
        # response side (the request side was set in before_model).
        span = otel_trace.get_current_span()
        if span.is_recording():
            if usage is not None:
                if usage.prompt_token_count is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_token_count)
                if usage.candidates_token_count is not None:
                    span.set_attribute("gen_ai.usage.output_tokens", usage.candidates_token_count)
            if llm_response.finish_reason is not None:
                span.set_attribute(
                    "gen_ai.response.finish_reasons", [str(llm_response.finish_reason)]
                )
            parts = (llm_response.content and llm_response.content.parts) or []
            reasoning = "".join(p.text or "" for p in parts if p.thought)
            content = "".join(p.text or "" for p in parts if p.text and not p.thought)
            tool_calls = [
                {
                    "id": p.function_call.id,
                    "type": "function",
                    "function": {
                        "name": p.function_call.name,
                        "arguments": json.dumps(p.function_call.args or {}),
                    },
                }
                for p in parts
                if p.function_call is not None
            ]
            span.set_attribute("gen_ai.response.tool_call_count", len(tool_calls))
            # The assistant reply, broken into its parts (reasoning first, then
            # content, then the calls it requested) so it reads top-to-bottom in
            # the span and sits visually after ``gen_ai.messages``.
            response_payload: dict[str, Any] = {"content": content}
            if reasoning:
                response_payload = {"reasoning": reasoning, **response_payload}
            if tool_calls:
                response_payload["tool_calls"] = tool_calls
            set_content_attribute(span, "gen_ai.response", response_payload)
        return None

    return after_model_callback


def make_before_tool_callback(
    context: ToolContext,
    settings: Settings,
    prompter: Prompter,
    callbacks: LoopCallbacks,
    session_id: str,
    *,
    allowed_tools: list[str] | None,
    permission_override: Any = None,
):
    async def before_tool_callback(*, tool, args, tool_context):
        nano_tool = tool.tool if isinstance(tool, NanoToolAdapter) else None
        if nano_tool is None:
            return {"result": f"Error: unknown tool '{tool.name}'."}
        name = nano_tool.name

        # ``allowed_tools`` is an execution restriction, not just prompt shaping:
        # a skill/subagent scoped to a subset must not run anything outside it,
        # even if the model emits an unadvertised call. Enforce before
        # permissions/hooks.
        if allowed_tools is not None and name not in allowed_tools:
            return {"result": f"Error: tool '{name}' is not permitted in this context."}

        # The model streamed argument JSON that wasn't parseable; the lenient
        # parser (see patches.py) marked it so the call fails softly here,
        # exactly like the old loop's JSON-decode error path.
        json_error = invalid_json_error(args)
        if json_error is not None:
            return {"result": f"Error: tool arguments were not valid JSON: {json_error}"}

        # MCP tools carry raw JSON Schema and receive the arg dict directly;
        # built-in tools validate against their Pydantic model first.
        if nano_tool.reads_raw_args:
            args_model: Any = args
        else:
            try:
                args_model = nano_tool.input_schema.model_validate(args)
            except ValidationError as exc:
                return {"result": f"Error: invalid arguments for {name}: {exc}"}

        # A permission_override replaces the whole policy (rules + mode + prompt)
        # — used by the memory-extraction fork to confine writes to the memory
        # dir. Prompts stay serialized across ADK's parallel tool tasks via the
        # permission manager's module-level lock.
        if permission_override is not None:
            decision = await permission_override(nano_tool, args_model, context)
        else:
            decision = await has_permission_to_use_tool(
                nano_tool, args_model, context, settings, prompter
            )
        if decision.behavior != "allow":
            log.info("tool %s denied: %s", name, decision.reason or "denied")
            if callbacks.on_tool_denied:
                callbacks.on_tool_denied(name, decision.reason)
            return {"result": f"Permission denied: {decision.reason or 'denied'}"}

        # PreToolUse hooks get the final say: exit code 2 denies the call.
        pre = await execute_hooks(
            HookEvent.PRE_TOOL_USE,
            session_id=session_id,
            cwd=context.cwd,
            tool_name=name,
            tool_input=args,
        )
        if pre.blocked:
            log.info("tool %s blocked by hook: %s", name, pre.block_reason)
            if callbacks.on_tool_denied:
                callbacks.on_tool_denied(name, pre.block_reason)
            return {"result": f"Blocked by hook: {pre.block_reason}"}

        return None

    return before_tool_callback


def make_after_tool_callback(context: ToolContext, session_id: str):
    async def after_tool_callback(*, tool, args, tool_context, tool_response):
        # PostToolUse hooks may append context the model sees with the result.
        # Skip fixed results injected by before_tool_callback (denials/errors):
        # those carry a dict response and never ran the tool.
        if not isinstance(tool_response, str):
            return None
        post = await execute_hooks(
            HookEvent.POST_TOOL_USE,
            session_id=session_id,
            cwd=context.cwd,
            tool_name=tool.name,
            tool_input=args,
            tool_response=tool_response,
        )
        if post.context_text:
            return {"result": f"{tool_response}\n[hook] {post.context_text}"}
        return None

    return after_tool_callback


def make_on_tool_error_callback():
    async def on_tool_error_callback(*, tool, args, tool_context, error):
        # ADK raises when the model calls a name outside tools_dict; the old
        # loop answered with an error string the model could recover from.
        # Distinguish a known-but-unadvertised tool from a hallucinated one.
        if isinstance(error, ValueError) and "not found" in str(error):
            if get_tool(tool.name) is not None:
                return {"result": f"Error: tool '{tool.name}' is not permitted in this context."}
            return {"result": f"Error: unknown tool '{tool.name}'."}
        return None  # anything else re-raises

    return on_tool_error_callback
