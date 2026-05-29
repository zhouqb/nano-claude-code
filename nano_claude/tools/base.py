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
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from nano_claude.permissions.modes import PermissionMode


@dataclass
class ToolContext:
    cwd: str
    cancel_event: asyncio.Event
    permission_mode: PermissionMode


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
