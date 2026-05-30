"""Layer 4 — Context Collapse.

A **read-time projection over the full history**, driven by a **commit log**.
Spans of consecutive read/search operations (Read / Grep / GlobTool turns) are
summarized into a single placeholder; the underlying messages are *not* deleted
from the canonical store — instead a ``CollapseCommit`` records the span's
boundaries + summary, and ``project_view`` replays the commit log on every turn
to splice each span down to its placeholder. That projection is what makes a
collapse persist across turns without mutating the transcript.

Mirrors Claude Code's ``contextCollapse`` (codename *marble-origami*), trimmed
to one mechanism. When enabled, collapse engages at ``collapse_commit_threshold``
(90% of the window) and commits spans until the *projected view* is back under
that threshold, **suppressing Layer 5** only when it actually gets there. If it
runs out of spans while still over, it reports ``exhausted`` and the pipeline
falls through to auto-compact — so an oversized request never goes out with
Layer 5 skipped.

Off by default (``AgentConfig.context_collapse``). Deliberately *in-memory only*
for now: commits live on ``LoopState`` for the session. Persisting the commit
log to JSONL so collapses survive ``--resume`` is a noted follow-up — see the
PR description.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import litellm

from nano_claude.compaction.thresholds import collapse_commit_threshold
from nano_claude.compaction.token_counter import estimate_message_tokens

if TYPE_CHECKING:
    from nano_claude.agent.types import AgentConfig, LoopState

# Pure read/search tools — output is bulky and re-derivable, safe to collapse.
READ_SEARCH_TOOLS = {"Read", "Grep", "GlobTool"}
# A span must be at least this many turns to be worth collapsing.
MIN_SPAN_TURNS = 2

SpanSummarizer = Callable[[list[dict], "AgentConfig"], Awaitable[str]]


@dataclass
class CollapseCommit:
    collapse_id: str
    first_id: str  # tool_call_id issued at the start of the span
    last_id: str  # tool_call_id of the last result in the span
    summary: str


@dataclass
class CollapseState:
    commits: list[CollapseCommit] = field(default_factory=list)


@dataclass
class _Span:
    first_id: str
    last_id: str
    messages: list[dict]


@dataclass
class CollapseResult:
    messages: list[dict]
    committed: bool  # collapsed a span this turn
    exhausted: bool  # over threshold but no span left to collapse


def reset_collapse(state: LoopState) -> None:
    """Drop all commits (e.g. on /clear or a conversation rewind)."""
    if state.collapse is not None:
        state.collapse.commits.clear()


def _name_by_id(messages: list[dict]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                out[tc.get("id")] = (tc.get("function") or {}).get("name")
    return out


def _is_read_search_turn(msg: dict, name_by_id: dict[str, str | None]) -> bool:
    if msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return False
    return all(name_by_id.get(tc.get("id")) in READ_SEARCH_TOOLS for tc in msg["tool_calls"])


def _splice(messages: list[dict], commit: CollapseCommit) -> list[dict]:
    """Replace the span [first_id .. last_id] with the commit's placeholder.

    Idempotent: once spliced, ``first_id`` is gone, so re-applying is a no-op.
    """
    start = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role") == "assistant"
            and any(tc.get("id") == commit.first_id for tc in (m.get("tool_calls") or []))
        ),
        None,
    )
    if start is None:
        return messages
    end = next(
        (
            i
            for i, m in enumerate(messages)
            if m.get("role") == "tool" and m.get("tool_call_id") == commit.last_id
        ),
        None,
    )
    if end is None or end < start:
        return messages
    placeholder = {"role": "assistant", "content": f"[collapsed earlier work: {commit.summary}]"}
    return messages[:start] + [placeholder] + messages[end + 1 :]


def project_view(messages: list[dict], state: CollapseState | None) -> list[dict]:
    """Replay the commit log to produce the collapsed view. Pure; no model call."""
    if state is None or not state.commits:
        return messages
    out = messages
    for commit in state.commits:
        out = _splice(out, commit)
    return out


def _find_span(messages: list[dict]) -> _Span | None:
    """Oldest maximal run of >= MIN_SPAN_TURNS consecutive read/search turns.

    A turn is the assistant tool-call message plus its tool results. The run
    breaks at any non-(read/search-turn) message. Runs on the already-projected
    view, so previously-collapsed spans are absent and never re-found.
    """
    name_by_id = _name_by_id(messages)
    i = 0
    n = len(messages)
    while i < n:
        if not _is_read_search_turn(messages[i], name_by_id):
            i += 1
            continue
        run_start = i
        turns = 0
        first_id = messages[i]["tool_calls"][0]["id"]
        last_id = first_id
        # Walk forward over consecutive read/search turns + their results.
        while i < n and _is_read_search_turn(messages[i], name_by_id):
            call_ids = {tc["id"] for tc in messages[i]["tool_calls"]}
            last_id = messages[i]["tool_calls"][-1]["id"]
            i += 1
            # consume this turn's tool results
            while (
                i < n
                and messages[i].get("role") == "tool"
                and messages[i].get("tool_call_id") in call_ids
            ):
                last_id = messages[i]["tool_call_id"]
                i += 1
            turns += 1
        if turns >= MIN_SPAN_TURNS:
            return _Span(first_id, last_id, messages[run_start:i])
    return None


async def _default_summarize(span_messages: list[dict], config: AgentConfig) -> str:
    """Summarize a span of read/search activity into one line (streamed)."""
    prompt = (
        "Summarize the following sequence of file reads and searches into a few "
        "sentences: what was looked at and what was found. Be specific about file "
        "paths and key findings; this replaces the raw output."
    )
    messages = [*span_messages, {"role": "user", "content": prompt}]
    response = await litellm.acompletion(model=config.model, messages=messages, stream=True)
    parts: list[str] = []
    async for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        content = getattr(choices[0].delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts).strip() or "(read/search activity)"


def _persist(commit: CollapseCommit, state: LoopState) -> None:
    """Forward-compatible persistence hook: writes the commit to the transcript
    once storage grows an ``append_collapse_boundary`` method. Until then collapse
    is in-memory only (commits don't survive ``--resume``). See PR notes."""
    if state.storage is not None and hasattr(state.storage, "append_collapse_boundary"):
        state.storage.append_collapse_boundary(
            collapse_id=commit.collapse_id,
            first_id=commit.first_id,
            last_id=commit.last_id,
            summary=commit.summary,
        )


async def apply_collapses_if_needed(
    messages: list[dict],
    state: LoopState,
    config: AgentConfig,
    *,
    summarize: SpanSummarizer | None = None,
) -> CollapseResult:
    """Project existing commits and collapse spans until the view fits.

    Headroom is recomputed from the *projected view* after every commit (it
    shrinks as spans collapse; ``last_input_tokens`` can't see this turn's
    collapses). We keep committing until the view is back under the threshold or
    no collapsible span remains. ``exhausted=True`` means we're still over the
    threshold with nothing left to collapse — the pipeline then falls through to
    Layer 5 so an oversized request never goes out with auto-compact skipped.
    """
    if state.collapse is None:
        state.collapse = CollapseState()
    summarize = summarize or _default_summarize

    threshold = collapse_commit_threshold(config.context_window)
    view = project_view(messages, state.collapse)
    if estimate_message_tokens(view) < threshold:
        return CollapseResult(view, committed=False, exhausted=False)

    committed = False
    while estimate_message_tokens(view) >= threshold:
        span = _find_span(view)
        if span is None:
            # Still over, but nothing left to collapse — Layer 5 takes over.
            return CollapseResult(view, committed=committed, exhausted=True)
        summary = await summarize(span.messages, config)
        commit = CollapseCommit(uuid.uuid4().hex[:16], span.first_id, span.last_id, summary)
        state.collapse.commits.append(commit)
        _persist(commit, state)
        committed = True
        view = project_view(messages, state.collapse)

    return CollapseResult(view, committed=committed, exhausted=False)
