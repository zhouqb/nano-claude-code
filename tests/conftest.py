"""Shared test fixtures and fakes for the streaming agent loop."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace


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
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
        usage=None,
    )


def usage_chunk(prompt_tokens: int, completion_tokens: int):
    """A final chunk carrying token usage and no content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=None,
        ),
    )


def tool_call_chunk(index: int, call_id: str, name: str, arguments: str):
    """A streaming chunk carrying an OpenAI-style tool_call delta."""
    tc = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))],
        usage=None,
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
