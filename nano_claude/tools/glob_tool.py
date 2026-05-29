"""Glob tool: find files by glob pattern, newest first."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult
from nano_claude.tools.overflow import save_overflow, truncation_note

MAX_RESULTS = 1000


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/*.ts'.")
    path: str | None = Field(
        default=None, description="Directory to search in. Defaults to the cwd."
    )


class GlobTool(Tool):
    name = "GlobTool"
    description = (
        "Find files matching a glob pattern (e.g. '**/*.py'). Returns matching "
        "paths sorted by modification time, newest first."
    )
    input_schema = GlobInput

    async def check_permissions(self, args: GlobInput, context: ToolContext) -> PermissionDecision:
        return PermissionDecision(behavior="allow")

    async def call(self, args: GlobInput, context: ToolContext) -> ToolResult:
        base = Path(args.path) if args.path else Path(context.cwd)
        if not base.is_absolute():
            base = Path(context.cwd) / base
        if not base.is_dir():
            return ToolResult.fail(f"Search path is not a directory: {base}")

        matches = [p for p in base.glob(args.pattern) if p.is_file()]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return ToolResult(output="No files matched.")

        out = "\n".join(str(p) for p in matches[:MAX_RESULTS])
        if len(matches) > MAX_RESULTS:
            full = "\n".join(str(p) for p in matches)
            spill = save_overflow(full, "Glob", context)
            out += truncation_note(spill, shown=MAX_RESULTS, total=len(matches), unit="matches")
        return ToolResult(output=out)
