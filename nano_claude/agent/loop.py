"""Core streaming query loop.

Phase 2 scope: the loop now advertises tools to the model, accumulates
streamed ``tool_calls``, gates each call through the permission manager, and
dispatches the allowed ones concurrently, appending one ``tool`` message per
call before looping again. The loop ends when the model returns no tool calls.

Permission *prompts* are resolved sequentially (so two ``ask`` tools in one
turn don't fight over the terminal); the actual tool *execution* is concurrent.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import litellm
from pydantic import ValidationError

from nano_claude.agent.types import AgentConfig, LoopResult, LoopState, StopReason
from nano_claude.compaction.pipeline import run_context_management
from nano_claude.compaction.token_counter import record_input_tokens
from nano_claude.extensibility.hooks import HookEvent, execute_hooks
from nano_claude.permissions.manager import (
    Prompter,
    PromptOutcome,
    has_permission_to_use_tool,
)
from nano_claude.permissions.settings import Settings
from nano_claude.session.storage import session_output_dir
from nano_claude.tools.base import ToolContext, ToolResult
from nano_claude.tools.registry import get_tool, get_tools

# Status codes worth retrying (rate limits, transient upstream errors).
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
MAX_RETRIES = 3
BASE_DELAY_S = 1.0

# Callback invoked with each text delta as it streams in.
TextCallback = Callable[[str], None]


@dataclass
class LoopCallbacks:
    """Optional display hooks; the REPL wires these to rich output."""

    on_text: TextCallback | None = None
    on_assistant_start: Callable[[], None] | None = None
    on_tool_start: Callable[[str, dict], None] | None = None
    on_tool_end: Callable[[str, ToolResult], None] | None = None
    on_tool_denied: Callable[[str, str], None] | None = None
    on_compact: Callable[[], None] | None = None
    on_compact_disabled: Callable[[], None] | None = None
    on_context_warning: Callable[[], None] | None = None
    on_snip: Callable[[int], None] | None = None
    on_collapse: Callable[[], None] | None = None


async def _deny_all_prompter(tool, args, prompt_text) -> PromptOutcome:
    """Fallback prompter used when none is supplied (non-interactive contexts)."""
    return PromptOutcome.DENY_ONCE


async def _call_with_retry(make_call: Callable[[], Awaitable]):
    """Invoke a streaming completion with exponential-backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            return await make_call()
        except litellm.exceptions.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY_S * (2**attempt))
        except litellm.exceptions.APIError as exc:
            status = getattr(exc, "status_code", 0)
            if status not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(BASE_DELAY_S * (2**attempt))


def _merge_tool_call_deltas(acc: list[dict], deltas) -> None:
    """Merge OpenAI-style streamed tool_call deltas (keyed by index)."""
    for d in deltas:
        idx = getattr(d, "index", 0) or 0
        while len(acc) <= idx:
            acc.append({"id": None, "name": None, "arguments": ""})
        slot = acc[idx]
        if getattr(d, "id", None):
            slot["id"] = d.id
        fn = getattr(d, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["arguments"] += fn.arguments


def _to_assistant_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Render accumulated calls into the OpenAI assistant-message shape."""
    return [
        {
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"},
        }
        for tc in tool_calls
    ]


# A resolved plan for one tool call: either ready-to-run or a fixed result string.
@dataclass
class _CallPlan:
    tool: Any | None = None
    args_model: Any | None = None
    args_dict: dict | None = None
    fixed_content: str | None = None  # set when the call won't run (error/denied)


async def _resolve_call(
    tc: dict,
    context: ToolContext,
    settings: Settings,
    prompter: Prompter,
    callbacks: LoopCallbacks,
    session_id: str,
    allowed_tools: list[str] | None,
) -> _CallPlan:
    """Validate args and resolve permissions for a single tool call."""
    name = tc.get("name")
    tool = get_tool(name) if name else None
    if tool is None:
        return _CallPlan(fixed_content=f"Error: unknown tool '{name}'.")

    # ``allowed_tools`` is an execution restriction, not just prompt shaping: a
    # skill/subagent scoped to a subset must not run anything outside it, even if
    # the model emits an unadvertised call. Enforce before permissions/hooks.
    if allowed_tools is not None and name not in allowed_tools:
        return _CallPlan(fixed_content=f"Error: tool '{name}' is not permitted in this context.")

    try:
        parsed = json.loads(tc.get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return _CallPlan(fixed_content=f"Error: tool arguments were not valid JSON: {exc}")

    # MCP tools carry raw JSON Schema and receive the arg dict directly; built-in
    # tools validate against their Pydantic model first.
    if tool.reads_raw_args:
        args_model: Any = parsed
    else:
        try:
            args_model = tool.input_schema.model_validate(parsed)
        except ValidationError as exc:
            return _CallPlan(fixed_content=f"Error: invalid arguments for {name}: {exc}")

    decision = await has_permission_to_use_tool(tool, args_model, context, settings, prompter)
    if decision.behavior != "allow":
        if callbacks.on_tool_denied:
            callbacks.on_tool_denied(name, decision.reason)
        return _CallPlan(fixed_content=f"Permission denied: {decision.reason or 'denied'}")

    # PreToolUse hooks get the final say: exit code 2 denies the call.
    pre = await execute_hooks(
        HookEvent.PRE_TOOL_USE,
        session_id=session_id,
        cwd=context.cwd,
        tool_name=name,
        tool_input=parsed,
    )
    if pre.blocked:
        if callbacks.on_tool_denied:
            callbacks.on_tool_denied(name, pre.block_reason)
        return _CallPlan(fixed_content=f"Blocked by hook: {pre.block_reason}")

    return _CallPlan(tool=tool, args_model=args_model, args_dict=parsed)


async def _run_call(
    plan: _CallPlan, context: ToolContext, callbacks: LoopCallbacks, session_id: str
) -> str:
    """Execute a resolved call (or return its fixed content)."""
    if plan.fixed_content is not None:
        return plan.fixed_content
    if context.cancel_event.is_set():
        return "[Interrupted]"

    if callbacks.on_tool_start:
        callbacks.on_tool_start(plan.tool.name, plan.args_dict or {})
    try:
        result = await plan.tool.call(plan.args_model, context)
    except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
        result = ToolResult.fail(f"Tool raised an exception: {exc}")
    if callbacks.on_tool_end:
        callbacks.on_tool_end(plan.tool.name, result)

    # PostToolUse hooks may append context the model sees with the result.
    post = await execute_hooks(
        HookEvent.POST_TOOL_USE,
        session_id=session_id,
        cwd=context.cwd,
        tool_name=plan.tool.name,
        tool_input=plan.args_dict or {},
        tool_response=result.output,
    )
    if post.context_text:
        return f"{result.output}\n[hook] {post.context_text}"
    return result.output


async def query_loop(
    state: LoopState,
    config: AgentConfig,
    *,
    settings: Settings | None = None,
    prompter: Prompter | None = None,
    on_text: TextCallback | None = None,
    callbacks: LoopCallbacks | None = None,
    allowed_tools: list[str] | None = None,
) -> LoopResult:
    """Run the agent loop until the model stops requesting tools.

    ``allowed_tools`` restricts the advertised tool set for this invocation
    (a skill's ``allowed-tools``, or a subagent's tool subset). ``None`` means
    the full registry for the current permission mode.
    """
    settings = settings or Settings()
    prompter = prompter or _deny_all_prompter
    callbacks = callbacks or LoopCallbacks()
    if on_text is not None:
        callbacks.on_text = on_text

    session_id = state.storage.session_id if state.storage is not None else ""
    tools = get_tools(config.permission_mode)
    if allowed_tools is not None:
        tools = [t for t in tools if t.name in allowed_tools]
    tool_schemas = [t.to_api_schema() for t in tools]
    context = ToolContext(
        cwd=config.cwd,
        cancel_event=state.cancel_event,
        permission_mode=config.permission_mode,
        output_dir=session_output_dir(state.storage),
        read_file_state=state.read_file_state,
    )

    def record(message: dict) -> None:
        """Append a message to state and persist it (for crash recovery)."""
        state.messages.append(message)
        if message.get("role") == "assistant":
            state.last_assistant_at = time.time()  # Layer 3 microcompact time gate
        if state.storage is not None:
            state.storage.append_message(message)

    while True:
        if state.cancel_event.is_set():
            return LoopResult(StopReason.ABORTED, state.turn_count, "")
        if state.turn_count >= config.max_turns:
            return LoopResult(StopReason.MAX_TURNS, state.turn_count, "")

        # Run the compaction pipeline and send its derived view to the model.
        # state.messages stays canonical (storage + scrollback); the view is
        # what the model sees this turn. Tool results / assistant turns below
        # are recorded into state.messages, so next iteration re-derives a
        # fresh view from the updated canonical store.
        view = await run_context_management(state, config, callbacks)
        if view.blocked:
            notice = "Context is full. Run /compact to continue."
            return LoopResult(StopReason.BLOCKED, state.turn_count, notice)

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        last_chunk = None
        started = False

        response = await _call_with_retry(
            lambda v=view: litellm.acompletion(
                model=config.model,
                messages=v.messages,
                tools=tool_schemas or None,
                stream=True,
                stream_options={"include_usage": True},
            )
        )

        async for chunk in response:
            if state.cancel_event.is_set():
                return LoopResult(StopReason.ABORTED, state.turn_count, "".join(text_parts))
            last_chunk = chunk
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                if not started and callbacks.on_assistant_start:
                    callbacks.on_assistant_start()
                started = True
                text_parts.append(content)
                if callbacks.on_text:
                    callbacks.on_text(content)
            tc_deltas = getattr(delta, "tool_calls", None)
            if tc_deltas:
                _merge_tool_call_deltas(tool_calls, tc_deltas)

        if last_chunk is not None:
            state.token_usage.update_from_litellm(last_chunk)
            # Record the API-reported input-token count — the current-context
            # signal the pipeline's thresholds key off next turn.
            record_input_tokens(state, last_chunk)
        state.turn_count += 1

        final_text = "".join(text_parts)

        if not tool_calls:
            record({"role": "assistant", "content": final_text})
            return LoopResult(StopReason.COMPLETED, state.turn_count, final_text)

        # Record the assistant turn (text + the calls it requested).
        record(
            {
                "role": "assistant",
                "content": final_text or None,
                "tool_calls": _to_assistant_tool_calls(tool_calls),
            }
        )

        # Resolve permissions sequentially, then execute allowed calls concurrently.
        plans = [
            await _resolve_call(tc, context, settings, prompter, callbacks, session_id, allowed_tools)
            for tc in tool_calls
        ]
        contents = await asyncio.gather(
            *(_run_call(plan, context, callbacks, session_id) for plan in plans)
        )

        for tc, content in zip(tool_calls, contents, strict=True):
            record({"role": "tool", "tool_call_id": tc["id"], "content": content})
        # Loop continues: the model sees the tool results next iteration.
