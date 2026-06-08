"""Tool base class and shared value types.

A ``Tool`` exposes a Pydantic ``input_schema`` (used both to validate model
output and to advertise the tool to the model via ``to_api_schema``), an async
``call`` that does the work, and ``check_permissions`` returning a
``PermissionDecision``. The central permission manager (Phase 2b) consults that
decision before any ``call`` runs.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from nano_claude.permissions.modes import PermissionMode


@dataclass
class FileReadSnapshot:
    content: str
    timestamp: float
    offset: int | None = None
    limit: int | None = None
    is_partial_view: bool = False


@dataclass
class ToolContext:
    cwd: str
    cancel_event: asyncio.Event
    permission_mode: PermissionMode
    # Directory where tools spill truncated output. Set by the loop to the
    # session's outputs folder; None falls back to a temp dir (see overflow.py).
    output_dir: Path | None = None
    # Claude Code requires a file to be read before an existing file is edited
    # or overwritten, and rejects stale writes if the file changed since then.
    read_file_state: dict[str, FileReadSnapshot] = field(default_factory=dict)
    # Threaded through so the Task tool can spawn a properly-configured subagent
    # loop (model inheritance, cost roll-up, shared permission path). Typed as
    # ``Any`` to avoid importing the loop/permission types here.
    parent_model: str = ""
    # Reasoning effort the parent loop runs with, so a spawned subagent inherits
    # the same thinking setting. None ⇒ provider default.
    parent_reasoning_effort: str | None = None
    token_usage_sink: Any = None  # parent TokenUsage; subagents merge cost into it
    settings: Any = None  # permission Settings, shared with subagents
    prompter: Any = None  # permission prompter, shared with subagents
    # Shared reference to the session's TodoWrite list (LoopState.todos). The
    # TodoWrite tool mutates it in place; None where there is no todo store.
    todos: list[dict[str, Any]] | None = None


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    error: str | None = None

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(output=error, is_error=True, error=error)


@dataclass
class PermissionDecision:
    behavior: Literal["allow", "deny", "ask"]
    reason: str = ""
    prompt: str = ""  # shown to the user when behavior == "ask"


class Tool(ABC):
    name: str
    description: str
    input_schema: type[BaseModel]
    # MCP tools carry a raw JSON Schema instead of a Pydantic model and receive
    # ``dict`` args. When True, the loop skips Pydantic validation and passes the
    # raw arg dict straight to ``call``/``check_permissions``.
    reads_raw_args: bool = False

    @abstractmethod
    async def call(self, args: BaseModel, context: ToolContext) -> ToolResult: ...

    @abstractmethod
    async def check_permissions(
        self, args: BaseModel, context: ToolContext
    ) -> PermissionDecision: ...

    def to_api_schema(self) -> dict[str, Any]:
        """Convert the Pydantic schema to an OpenAI-style tool definition.

        This is LiteLLM's common format; it normalises to each provider's native
        tool schema transparently.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }
