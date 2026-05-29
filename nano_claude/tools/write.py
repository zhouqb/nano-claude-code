"""Write tool: create or overwrite a file with the given contents."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult


class WriteInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to write.")
    content: str = Field(description="The full contents to write to the file.")


class WriteTool(Tool):
    name = "Write"
    description = (
        "Write content to a file, creating it (and parent directories) or "
        "overwriting it if it already exists."
    )
    input_schema = WriteInput

    async def check_permissions(self, args: WriteInput, context: ToolContext) -> PermissionDecision:
        path = self._resolve(args.file_path, context)
        verb = "Overwrite" if path.exists() else "Create"
        return PermissionDecision(behavior="ask", prompt=f"{verb} file {path}?")

    @staticmethod
    def _resolve(file_path: str, context: ToolContext) -> Path:
        path = Path(file_path)
        return path if path.is_absolute() else Path(context.cwd) / path

    async def call(self, args: WriteInput, context: ToolContext) -> ToolResult:
        path = self._resolve(args.file_path, context)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(args.content)
        except OSError as exc:
            return ToolResult.fail(f"Could not write {path}: {exc}")

        line_count = args.content.count("\n") + 1 if args.content else 0
        return ToolResult(output=f"Wrote {line_count} line(s) to {path}")
