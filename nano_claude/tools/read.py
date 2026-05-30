"""Read tool: read a file's contents with cat -n style line numbers."""

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

DEFAULT_LIMIT = 2000


class ReadInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file to read.")
    offset: int = Field(default=0, description="0-based line number to start reading from.")
    limit: int = Field(default=DEFAULT_LIMIT, description="Maximum number of lines to read.")


class ReadTool(Tool):
    name = "Read"
    description = (
        "Read a file from the local filesystem. Returns the contents with "
        "1-based line numbers. Reads up to 2000 lines by default; use offset/limit "
        "to page through larger files."
    )
    input_schema = ReadInput

    async def check_permissions(self, args: ReadInput, context: ToolContext) -> PermissionDecision:
        return PermissionDecision(behavior="allow")

    async def call(self, args: ReadInput, context: ToolContext) -> ToolResult:
        path = Path(args.file_path)
        if not path.is_absolute():
            path = Path(context.cwd) / path
        if not path.exists():
            return ToolResult.fail(f"File not found: {path}")
        if path.is_dir():
            return ToolResult.fail(f"Path is a directory, not a file: {path}")

        try:
            async with aiofiles.open(path, encoding="utf-8", errors="replace") as f:
                content = await f.read()
        except OSError as exc:
            return ToolResult.fail(f"Could not read {path}: {exc}")

        lines = content.splitlines()
        start = max(args.offset, 0)
        selected = lines[start : start + args.limit]
        truncated = len(lines) > start + args.limit
        if start == 0 and not truncated:
            context.read_file_state[str(path)] = FileReadSnapshot(
                content=content,
                timestamp=path.stat().st_mtime,
                offset=None,
                limit=None,
                is_partial_view=False,
            )
        else:
            context.read_file_state[str(path)] = FileReadSnapshot(
                content="\n".join(selected),
                timestamp=path.stat().st_mtime,
                offset=start,
                limit=args.limit,
                is_partial_view=True,
            )
        if not selected:
            return ToolResult(output="(file is empty or offset past end of file)")

        numbered = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(selected, start=start + 1))
        if truncated:
            numbered += f"\n... ({len(lines) - start - args.limit} more lines truncated)"
        return ToolResult(output=numbered)
