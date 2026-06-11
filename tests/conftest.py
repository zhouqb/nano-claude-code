"""Shared test fixtures and fakes for the agent.

The LLM seam is unchanged from the pre-ADK days: tests monkeypatch
``litellm.acompletion``. The ADK driver's ``LiteLlm`` wrapper calls straight
through to it, so these fakes now exercise the *entire* production stack
(Runner → LlmAgent → LiteLlm → chunk aggregation) rather than a bespoke loop.

The chunk helpers therefore build real ``litellm`` response types — ADK's
stream parser type-checks against ``ModelResponseStream`` — but their names
and signatures are unchanged, so tests read the same as before. The same
fakes also serve the modules that still call ``litellm.acompletion``
directly (compactor, memory recall/extract).
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


@pytest.fixture(autouse=True)
def _route_adk_through_litellm(monkeypatch):
    """Keep ``monkeypatch.setattr(litellm, "acompletion", ...)`` effective.

    ADK's ``lite_llm`` module lazily binds ``litellm.acompletion`` into its own
    namespace on first use, after which patching the ``litellm`` module no
    longer reaches it. Re-route ADK's bound name through the ``litellm`` module
    attribute at call time so the historical seam keeps working everywhere.
    """
    import litellm
    from google.adk.models import lite_llm as _adk_lite_llm

    async def _delegate(**kwargs):
        return await litellm.acompletion(**kwargs)

    _adk_lite_llm._ensure_litellm_imported()
    monkeypatch.setattr(_adk_lite_llm, "acompletion", _delegate)


class FakeStream:
    """Minimal async-iterable mimicking a LiteLLM streaming response."""

    def __init__(self, chunks: Iterable):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def text_chunk(content: str):
    """A streaming chunk carrying a content delta (OpenAI-normalised shape)."""
    return ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=content))])


def usage_chunk(prompt_tokens: int, completion_tokens: int):
    """A final chunk carrying token usage and no content."""
    chunk = ModelResponseStream(choices=[StreamingChoices(delta=Delta(content=None))])
    chunk.usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return chunk


def tool_call_chunk(index: int, call_id: str, name: str, arguments: str):
    """A streaming chunk carrying an OpenAI-style tool_call delta."""
    tc = ChatCompletionDeltaToolCall(
        index=index,
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )
    return ModelResponseStream(
        choices=[StreamingChoices(delta=Delta(content=None, tool_calls=[tc]))]
    )


def make_acompletion(chunks):
    """Build an async ``acompletion`` stand-in returning ``FakeStream(chunks)``."""

    async def _acompletion(*args, **kwargs):
        return FakeStream(chunks)

    return _acompletion


def make_sequential_acompletion(streams: Iterable):
    """Return successive ``FakeStream``s on each call (for multi-turn loops)."""
    it = iter([FakeStream(s) for s in streams])

    async def _acompletion(*args, **kwargs):
        return next(it)

    return _acompletion
