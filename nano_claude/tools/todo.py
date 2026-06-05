"""The TodoWrite tool — a structured, session-scoped task checklist.

A faithful port of Claude Code's ``TodoWrite``. The model rewrites the whole
list on every call (there is no incremental edit); the tool stores it on the
:class:`~nano_claude.agent.types.LoopState` (threaded in via
``ToolContext.todos``) so the REPL can render it and the loop can nudge the
model when the list goes stale.

Like Claude Code, when *every* item is ``completed`` the stored list is cleared
— the work is done, so there is nothing left to track or to remind about. The
tool result still reports success and the UI still renders the all-complete
list once (it draws from the call's input, not the cleared store).

The list is keyed implicitly by ``LoopState``: a subagent runs with its own
state, so it tracks its own checklist, matching Claude Code's per-agent keying.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult

TodoStatus = Literal["pending", "in_progress", "completed"]

# Guidance handed to the model via the tool schema. Mirrors Claude Code's
# TodoWrite prompt: when (and when not) to use it, the one-in_progress rule, and
# the content/activeForm requirement.
_DESCRIPTION = """\
Use this tool to create and manage a structured task list for your current coding session. \
This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. \
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. \
In this case you are better off just doing the task directly.

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

   **IMPORTANT**: Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done (e.g., "Run tests", "Build the project")
   - activeForm: The present continuous form shown during execution (e.g., "Running tests", "Building the project")

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names
   - Always provide both forms:
     - content: "Fix authentication bug"
     - activeForm: "Fixing authentication bug"

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness \
and ensures you complete all requirements successfully."""

# The fixed result string Claude Code returns; some models key off the exact
# wording, so keep it verbatim.
_SUCCESS = (
    "Todos have been modified successfully. Ensure that you continue to use the "
    "todo list to track your progress. Please proceed with the current tasks if applicable"
)


class TodoItem(BaseModel):
    content: str = Field(
        min_length=1, description="The imperative form of the task (e.g. 'Run tests')."
    )
    status: TodoStatus = Field(description="One of: pending, in_progress, completed.")
    activeForm: str = Field(
        min_length=1,
        description="The present-continuous form shown while active (e.g. 'Running tests').",
    )


class TodoWriteInput(BaseModel):
    todos: list[TodoItem] = Field(description="The updated todo list (the full list, not a delta).")


class TodoWriteTool(Tool):
    name = "TodoWrite"
    description = _DESCRIPTION
    input_schema = TodoWriteInput

    async def check_permissions(
        self, args: TodoWriteInput, context: ToolContext
    ) -> PermissionDecision:
        # Managing an in-memory checklist touches nothing external; always allow.
        return PermissionDecision(behavior="allow")

    async def call(self, args: TodoWriteInput, context: ToolContext) -> ToolResult:
        if context.todos is None:
            return ToolResult.fail("TodoWrite is unavailable in this context (no todo store).")

        items = [item.model_dump() for item in args.todos]
        all_done = bool(items) and all(item["status"] == "completed" for item in items)

        # Replace the shared list in place so LoopState.todos stays the same
        # object. When everything is done there is nothing left to track, so the
        # stored list is cleared (the UI still rendered the full list from the
        # call input on the way in).
        context.todos.clear()
        if not all_done:
            context.todos.extend(items)

        return ToolResult(output=_SUCCESS)
