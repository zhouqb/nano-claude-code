"""Core streaming query loop.

Phase 1 scope: a single streaming model call per turn, no tool dispatch. The
loop is shaped as the agentic ``while`` loop from the plan so that Phase 2 can
add tool collection + dispatch without restructuring. With no tools wired in,
the model never emits tool calls, so each loop resolves in one iteration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import litellm

from nano_claude.agent.types import AgentConfig, LoopResult, LoopState, StopReason

# Status codes worth retrying (rate limits, transient upstream errors).
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
MAX_RETRIES = 3
BASE_DELAY_S = 1.0

# Callback invoked with each text delta as it streams in.
TextCallback = Callable[[str], None]


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


async def query_loop(
    state: LoopState,
    config: AgentConfig,
    on_text: TextCallback | None = None,
) -> LoopResult:
    """Run the agent loop until the model stops requesting tools.

    In Phase 1 there are no tools, so this performs one streaming completion and
    returns ``StopReason.COMPLETED``.
    """
    while True:
        if state.cancel_event.is_set():
            return LoopResult(StopReason.ABORTED, state.turn_count, "")

        if state.turn_count >= config.max_turns:
            return LoopResult(StopReason.MAX_TURNS, state.turn_count, "")

        text_parts: list[str] = []
        last_chunk = None

        response = await _call_with_retry(
            lambda: litellm.acompletion(
                model=config.model,
                messages=state.messages,
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
                text_parts.append(content)
                if on_text is not None:
                    on_text(content)

        if last_chunk is not None:
            state.token_usage.update_from_litellm(last_chunk)
        state.turn_count += 1

        final_text = "".join(text_parts)
        state.messages.append({"role": "assistant", "content": final_text})

        # Phase 2 will collect tool_calls here and continue the loop when present.
        return LoopResult(StopReason.COMPLETED, state.turn_count, final_text)
