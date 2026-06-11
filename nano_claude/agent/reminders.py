"""TodoWrite staleness reminders, injected into the conversation pre-request.

Moved verbatim from the old streaming loop; the ADK driver's
``before_model_callback`` calls :func:`maybe_todo_reminder` each turn.
"""

from __future__ import annotations

from nano_claude.agent.types import LoopState
from nano_claude.tools.registry import get_tool

# TodoWrite staleness nudge: fire a reminder only when at least this many
# assistant turns have passed since the last TodoWrite call *and* since the last
# reminder. Values are Claude Code's TODO_REMINDER_CONFIG verbatim
# (TURNS_SINCE_WRITE / TURNS_BETWEEN_REMINDERS, both 10).
TODO_TURNS_SINCE_WRITE = 10
TODO_TURNS_BETWEEN_REMINDERS = 10
# Leading sentence of the injected reminder; also used to detect prior reminders
# when counting turns, so keep it in sync with ``build_todo_reminder``.
TODO_REMINDER_LEAD = "The TodoWrite tool hasn't been used recently."


def has_todowrite_call(msg: dict) -> bool:
    """True if an assistant message requested a TodoWrite tool call."""
    for tc in msg.get("tool_calls") or []:
        if (tc.get("function") or {}).get("name") == "TodoWrite":
            return True
    return False


def is_todo_reminder(msg: dict) -> bool:
    """True if a user message is a previously-injected TodoWrite staleness nudge."""
    content = msg.get("content")
    return isinstance(content, str) and TODO_REMINDER_LEAD in content


def todo_turn_counts(messages: list[dict]) -> tuple[int, int]:
    """Assistant turns since the last TodoWrite call and since the last reminder.

    Mirrors Claude Code's ``getTodoReminderTurnCounts``: walk backwards counting
    assistant turns; the TodoWrite turn and the reminder turn themselves are not
    counted. When TodoWrite/reminder was never seen, the count is the total
    number of assistant turns so far.
    """
    since_write = 0
    since_reminder = 0
    found_write = False
    found_reminder = False
    for msg in reversed(messages):
        role = msg.get("role")
        if role == "assistant":
            if not found_write and has_todowrite_call(msg):
                found_write = True
            if not found_write:
                since_write += 1
            if not found_reminder:
                since_reminder += 1
        elif role == "user" and not found_reminder and is_todo_reminder(msg):
            found_reminder = True
        if found_write and found_reminder:
            break
    return since_write, since_reminder


def build_todo_reminder(todos: list[dict]) -> dict:
    """Build the ``<system-reminder>`` user message nudging TodoWrite use.

    Wording follows Claude Code's ``messages.ts``, lightly grammar-corrected
    ("if it has become stale" vs. their "if has become stale").
    """
    body = (
        TODO_REMINDER_LEAD
        + " If you're working on tasks that would benefit from tracking progress, "
        "consider using the TodoWrite tool to track progress. Also consider cleaning "
        "up the todo list if it has become stale and no longer matches what you are "
        "working on. Only use it if it's relevant to the current work. This is just a "
        "gentle reminder - ignore if not applicable. Make sure that you NEVER mention "
        "this reminder to the user"
    )
    if todos:
        items = "\n".join(
            f"{i + 1}. [{t.get('status')}] {t.get('content')}" for i, t in enumerate(todos)
        )
        body += f"\n\nHere are the existing contents of your todo list:\n\n[{items}]"
    return {"role": "user", "content": f"<system-reminder>\n{body}\n</system-reminder>"}


def maybe_todo_reminder(state: LoopState, allowed_tools: list[str] | None) -> dict | None:
    """Return a staleness-nudge message if TodoWrite is available and overdue."""
    if get_tool("TodoWrite") is None:
        return None
    if allowed_tools is not None and "TodoWrite" not in allowed_tools:
        return None
    since_write, since_reminder = todo_turn_counts(state.messages)
    if since_write >= TODO_TURNS_SINCE_WRITE and since_reminder >= TODO_TURNS_BETWEEN_REMINDERS:
        return build_todo_reminder(state.todos)
    return None
