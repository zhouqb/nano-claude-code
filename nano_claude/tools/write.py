"""Write tool: create or overwrite a file with the given contents."""

from __future__ import annotations

from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from nano_claude.tools.base import (
    FileReadSnapshot,
    PermissionDecision,
    Tool,
    ToolContext,
    ToolResult,
)


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
        if path.exists():
            if not path.is_file():
                return ToolResult.fail(f"Path is not a file: {path}")
            stale_error = self._stale_write_error(path, context)
            if stale_error is not None:
                return ToolResult.fail(stale_error)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(args.content)
        except OSError as exc:
            return ToolResult.fail(f"Could not write {path}: {exc}")

        context.read_file_state[str(path)] = FileReadSnapshot(
            content=args.content,
            timestamp=path.stat().st_mtime,
            offset=None,
            limit=None,
            is_partial_view=False,
        )
        line_count = args.content.count("\n") + 1 if args.content else 0
        return ToolResult(output=f"Wrote {line_count} line(s) to {path}")

    @staticmethod
    def _stale_write_error(path: Path, context: ToolContext) -> str | None:
        # Same model as Edit: the existing file must have been read this session,
        # and must not have changed on disk since (keyed on mtime, so a partial
        # read still counts). See EditTool._stale_write_error for the rationale.
        snapshot = context.read_file_state.get(str(path))
        if snapshot is None:
            return "File has not been read yet. Read it first before writing to it."
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            return None
        if current_mtime != snapshot.timestamp:
            return "File has been modified since read. Read it again before attempting to write it."
        return None
