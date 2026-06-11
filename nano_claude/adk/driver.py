"""The turn driver: ``run_turn`` is the drop-in replacement for ``query_loop``.

ADK owns the protocol machinery the old loop hand-rolled (LiteLLM streaming,
tool-call delta merging, parallel dispatch); everything nano-specific rides in
the callbacks (:mod:`nano_claude.adk.callbacks`) and the tool adapter. The
contract is unchanged: OpenAI-format dicts in ``state.messages`` stay the
canonical history (recorded here as ADK events stream out), and the result is
a :class:`~nano_claude.agent.types.LoopResult`.

A fresh ``LlmAgent`` + ``Runner`` (with an in-memory ADK session) is built per
call — agents are lightweight pydantic objects, and per-turn construction is
what binds the turn's ``ToolContext``/``allowed_tools`` into the adapters and
lets ``/model`` switches take effect next turn. The ADK session is purely
transport; nothing reads it back (the session JSONL is written via
``record``, exactly like the old loop).
"""

from __future__ import annotations

import time
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from nano_claude.adk.callbacks import (
    TurnGates,
    make_after_model_callback,
    make_after_tool_callback,
    make_before_model_callback,
    make_before_tool_callback,
    make_on_tool_error_callback,
)
from nano_claude.adk.convert import event_to_messages
from nano_claude.adk.patches import apply as _apply_adk_patches
from nano_claude.adk.tool_adapter import NanoToolAdapter
from nano_claude.agent.types import AgentConfig, LoopCallbacks, LoopResult, LoopState, StopReason
from nano_claude.agent.types import TextCallback as TextCallback  # re-export for callers
from nano_claude.permissions.manager import Prompter, PromptOutcome
from nano_claude.permissions.settings import Settings
from nano_claude.session.storage import session_output_dir
from nano_claude.telemetry import log, tracer
from nano_claude.tools.base import ToolContext
from nano_claude.tools.registry import get_tools

_APP_NAME = "nano-claude"
_USER_ID = "local"

_apply_adk_patches()


async def _deny_all_prompter(tool, args, prompt_text) -> PromptOutcome:
    """Fallback prompter used when none is supplied (non-interactive contexts)."""
    return PromptOutcome.DENY_ONCE


def _build_model(config: AgentConfig) -> LiteLlm:
    """The model wrapper for this turn. Tests monkeypatch litellm.acompletion
    underneath it, so this stays a thin pass-through."""
    extra: dict[str, Any] = {"num_retries": 3}
    if config.reasoning_effort is not None:
        # litellm translates the unified ``reasoning_effort`` to the right
        # per-provider payload (OpenAI native / Anthropic + DeepSeek
        # ``thinking``); omit it entirely when unset.
        extra["reasoning_effort"] = config.reasoning_effort
    return LiteLlm(model=config.model, **extra)


def _last_user_content(state: LoopState) -> types.Content:
    """The newest user message, as the ``new_message`` ADK bookkeeping wants.

    The request the model actually sees is rebuilt from the pipeline view in
    ``before_model_callback``, so this only seeds the throwaway ADK session.
    """
    for msg in reversed(state.messages):
        if msg.get("role") == "user":
            return types.Content(role="user", parts=[types.Part(text=msg.get("content") or "")])
    return types.Content(role="user", parts=[types.Part(text="")])


def _reasoning_text(event) -> str:
    """Concatenated thought text from a model event (trace/log only)."""
    if event.content is None or not event.content.parts:
        return ""
    return "".join(p.text or "" for p in event.content.parts if p.thought)


async def run_turn(
    state: LoopState,
    config: AgentConfig,
    *,
    settings: Settings | None = None,
    prompter: Prompter | None = None,
    on_text: TextCallback | None = None,
    callbacks: LoopCallbacks | None = None,
    allowed_tools: list[str] | None = None,
    memory_prefetch: Any = None,
    permission_override: Any = None,
) -> LoopResult:
    """Run the agent until the model stops requesting tools.

    ``allowed_tools`` restricts the advertised tool set for this invocation
    (a skill's ``allowed-tools``, or a subagent's tool subset). ``None`` means
    the full registry for the current permission mode.

    ``memory_prefetch`` is an optional :class:`~nano_claude.memory.recall.MemoryPrefetch`
    fired for this user turn; the driver drains it (zero-wait) once it has
    settled and injects the surfaced memory files. Subagents pass ``None``.
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
    context = ToolContext(
        cwd=config.cwd,
        cancel_event=state.cancel_event,
        permission_mode=config.permission_mode,
        output_dir=session_output_dir(state.storage),
        read_file_state=state.read_file_state,
        # Carried so the Task tool can spawn a subagent that inherits the model,
        # rolls its cost up here, and shares the permission path.
        parent_model=config.model,
        parent_reasoning_effort=config.reasoning_effort,
        token_usage_sink=state.token_usage,
        settings=settings,
        prompter=prompter,
        todos=state.todos,  # TodoWrite mutates this list in place
    )

    def record(message: dict) -> None:
        """Append a message to state and persist it (for crash recovery)."""
        state.messages.append(message)
        if message.get("role") == "assistant":
            state.last_assistant_at = time.time()  # Layer 3 microcompact time gate
        if state.storage is not None:
            state.storage.append_message(message)

    gates = TurnGates()
    agent = LlmAgent(
        name="nano_claude",
        model=_build_model(config),
        instruction="",  # replaced wholesale by the view's system message
        tools=[
            NanoToolAdapter(
                t, context, on_tool_start=callbacks.on_tool_start, on_tool_end=callbacks.on_tool_end
            )
            for t in tools
        ],
        before_model_callback=make_before_model_callback(
            state,
            config,
            callbacks,
            gates,
            allowed_tools=allowed_tools,
            memory_prefetch=memory_prefetch,
            record=record,
        ),
        after_model_callback=make_after_model_callback(state, gates),
        before_tool_callback=make_before_tool_callback(
            context,
            settings,
            prompter,
            callbacks,
            session_id,
            allowed_tools=allowed_tools,
            permission_override=permission_override,
        ),
        after_tool_callback=make_after_tool_callback(context, session_id),
        on_tool_error_callback=make_on_tool_error_callback(),
    )

    session_service = InMemorySessionService()
    adk_session = await session_service.create_session(
        app_name=_APP_NAME, user_id=_USER_ID, session_id=session_id or "subagent"
    )
    runner = Runner(app_name=_APP_NAME, agent=agent, session_service=session_service)
    run_config = RunConfig(
        streaming_mode=StreamingMode.SSE,
        # Defensive backstop only; the cumulative gate in before_model_callback
        # (state.turn_count, which spans REPL turns) is the real limit.
        max_llm_calls=config.max_turns + 10,
    )

    turn_attrs: dict[str, Any] = {
        "nano_claude.model": config.model,
        "nano_claude.permission_mode": config.permission_mode.value,
        # Subagents run with no storage (empty session_id) — tag the span so a
        # nested subagent turn is distinguishable in traces.
        "nano_claude.subagent": session_id == "",
    }
    if allowed_tools is not None:
        turn_attrs["nano_claude.allowed_tools"] = len(allowed_tools)

    final_text = ""
    partial_parts: list[str] = []  # in-flight response text, for the ABORTED result
    with tracer.start_as_current_span("agent.turn", attributes=turn_attrs):
        events = runner.run_async(
            user_id=_USER_ID,
            session_id=adk_session.id,
            new_message=_last_user_content(state),
            run_config=run_config,
        )
        try:
            async for event in events:
                if state.cancel_event.is_set():
                    gates.reason = StopReason.ABORTED
                    break

                if event.partial:
                    # Streaming deltas: text goes to the display; reasoning is
                    # deliberately not shown (captured from the final response).
                    if event.content and event.content.parts:
                        delta = "".join(
                            p.text or "" for p in event.content.parts if p.text and not p.thought
                        )
                        if delta:
                            if not gates.assistant_started and callbacks.on_assistant_start:
                                callbacks.on_assistant_start()
                            gates.assistant_started = True
                            partial_parts.append(delta)
                            if callbacks.on_text:
                                callbacks.on_text(delta)
                    continue

                # Reasoning is captured for the trace + log file but deliberately
                # kept out of state.messages, so it is never sent back to the model.
                reasoning = _reasoning_text(event)
                if reasoning:
                    log.info("assistant reasoning (%d chars): %s", len(reasoning), reasoning)

                partial_parts.clear()  # response completed; nothing in flight
                recorded = event_to_messages(event)
                for msg in recorded:
                    record(msg)
                if event.content and event.content.role == "model":
                    if not recorded and gates.reason is None:
                        # A genuinely empty final response: keep the old loop's
                        # byte-format (an empty assistant message is recorded).
                        record({"role": "assistant", "content": ""})
                    if not event.get_function_calls():
                        final_text = next(
                            (
                                m.get("content") or ""
                                for m in reversed(recorded)
                                if m.get("role") == "assistant"
                            ),
                            "",
                        )
        finally:
            await events.aclose()

    if state.cancel_event.is_set() and gates.reason is None:
        gates.reason = StopReason.ABORTED
    if gates.reason is StopReason.ABORTED:
        # Match the old loop: an abort mid-stream returns the text so far.
        return LoopResult(StopReason.ABORTED, state.turn_count, "".join(partial_parts))
    if gates.reason is StopReason.MAX_TURNS:
        return LoopResult(StopReason.MAX_TURNS, state.turn_count, "")
    if gates.reason is StopReason.BLOCKED:
        return LoopResult(StopReason.BLOCKED, state.turn_count, gates.notice)

    log.info("turn complete (no tool calls) after %d iteration(s)", state.turn_count)
    return LoopResult(StopReason.COMPLETED, state.turn_count, final_text)
