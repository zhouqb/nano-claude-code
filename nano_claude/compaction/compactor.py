"""Conversation compaction: summarize the history and replace it.

On success the message list becomes ``[system?, summary, *recent_tail]`` where
the recent tail is kept API-valid (no orphan tool messages, no dangling tool
calls). On failure the failure counter is bumped for the circuit breaker.

The summary prompt is reproduced **verbatim** from Claude Code's
``src/services/compact/prompt.ts`` (``getCompactPrompt()`` with no custom
instructions = ``NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT + NO_TOOLS_TRAILER``).
That prompt asks the model to emit an ``<analysis>`` scratchpad followed by a
``<summary>`` block; :func:`format_compact_summary` mirrors CC's post-processing
to strip the scratchpad and unwrap the summary before it re-enters context.
"""

from __future__ import annotations

import re

import litellm

from nano_claude.agent.types import AgentConfig, LoopState
from nano_claude.session.restore import repair_messages

# How many of the most recent (non-system) messages to keep verbatim.
RECENT_MESSAGES_KEPT = 6

# --- Prompt (verbatim from Claude Code's compact/prompt.ts) -----------------

NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

BASE_COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>
"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)

# getCompactPrompt() with no custom instructions.
SUMMARY_PROMPT = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT + NO_TOOLS_TRAILER


def format_compact_summary(summary: str) -> str:
    """Strip the <analysis> scratchpad and unwrap <summary> (mirrors CC)."""
    # Drop the analysis section — a drafting scratchpad with no lasting value.
    formatted = re.sub(r"<analysis>[\s\S]*?</analysis>", "", summary)
    # Replace the <summary>...</summary> wrapper with a readable header.
    match = re.search(r"<summary>([\s\S]*?)</summary>", formatted)
    if match:
        content = (match.group(1) or "").strip()
        formatted = re.sub(
            r"<summary>[\s\S]*?</summary>",
            lambda _: f"Summary:\n{content}",
            formatted,
        )
    # Collapse runs of blank lines left behind.
    formatted = re.sub(r"\n\n+", "\n\n", formatted)
    return formatted.strip()


def build_compact_user_message(
    summary: str,
    *,
    transcript_path: str | None = None,
    recent_messages_preserved: bool = False,
    suppress_follow_up: bool = False,
) -> str:
    """Wrap a formatted summary into the continuation message (verbatim from CC's
    ``getCompactUserSummaryMessage`` in compact/prompt.ts; proactive-mode branch
    omitted)."""
    base = (
        "This session is being continued from a previous conversation that ran out of "
        "context. The summary below covers the earlier portion of the conversation.\n\n"
        f"{summary}"
    )

    if transcript_path:
        base += (
            "\n\nIf you need specific details from before compaction (like exact code "
            "snippets, error messages, or content you generated), read the full transcript "
            f"at: {transcript_path}"
        )

    if recent_messages_preserved:
        base += "\n\nRecent messages are preserved verbatim."

    if suppress_follow_up:
        base += (
            "\nContinue the conversation from where it left off without asking the user any "
            "further questions. Resume directly — do not acknowledge the summary, do not "
            'recap what was happening, do not preface with "I\'ll continue" or similar. Pick '
            "up the last task as if the break never happened."
        )

    return base


async def _summarize(state: LoopState, config: AgentConfig) -> str:
    """Ask the model to summarize the conversation (streamed, then joined)."""
    messages = [*state.messages, {"role": "user", "content": SUMMARY_PROMPT}]
    response = await litellm.acompletion(
        model=config.model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    async for chunk in response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        content = getattr(choices[0].delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts)


def _safe_recent_tail(messages: list[dict]) -> list[dict]:
    """Keep the last few non-system messages as a valid, self-contained tail."""
    non_system = [m for m in messages if m.get("role") != "system"]
    tail = non_system[-RECENT_MESSAGES_KEPT:]
    # Drop leading orphan tool results whose assistant call isn't in the tail.
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)
    # Repair any dangling tool_calls left at the end of the tail.
    return repair_messages(tail)


async def compact_conversation(state: LoopState, config: AgentConfig) -> bool:
    """Compact ``state.messages`` in place. Returns True on success."""
    pre_turn_count = state.turn_count
    try:
        raw = await _summarize(state, config)
    except Exception:  # noqa: BLE001 - any failure feeds the circuit breaker
        state.consecutive_compact_failures += 1
        return False

    summary = format_compact_summary(raw)
    if not summary:
        state.consecutive_compact_failures += 1
        return False

    system = next((m for m in state.messages if m.get("role") == "system"), None)
    tail = _safe_recent_tail(state.messages)

    transcript_path = str(state.storage.path) if state.storage is not None else None
    content = build_compact_user_message(
        summary,
        transcript_path=transcript_path,
        recent_messages_preserved=bool(tail),
        suppress_follow_up=True,
    )

    new_messages: list[dict] = []
    if system is not None:
        new_messages.append(system)
    new_messages.append({"role": "user", "content": content})
    new_messages.extend(tail)

    state.messages = new_messages
    state.last_input_tokens = 0
    state.consecutive_compact_failures = 0

    if state.storage is not None:
        state.storage.append_compact_boundary(summary=summary, pre_turn_count=pre_turn_count)
    return True
