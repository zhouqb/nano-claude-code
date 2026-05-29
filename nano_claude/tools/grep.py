"""Grep tool: search file contents with ripgrep, falling back to grep."""

from __future__ import annotations

import asyncio
import shutil

from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult
from nano_claude.tools.overflow import save_overflow, truncation_note

MAX_OUTPUT_BYTES = 60_000


class GrepInput(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str | None = Field(
        default=None, description="File or directory to search. Defaults to the cwd."
    )
    glob: str | None = Field(
        default=None, description="Optional glob to filter files, e.g. '*.py'."
    )


class GrepTool(Tool):
    name = "Grep"
    description = (
        "Search file contents for a regular expression. Uses ripgrep when "
        "available (falling back to grep). Returns matching lines with file:line "
        "prefixes."
    )
    input_schema = GrepInput

    async def check_permissions(self, args: GrepInput, context: ToolContext) -> PermissionDecision:
        return PermissionDecision(behavior="allow")

    def _build_command(self, args: GrepInput) -> list[str]:
        target = args.path or "."
        if shutil.which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--color", "never"]
            if args.glob:
                cmd += ["--glob", args.glob]
            cmd += ["--", args.pattern, target]
            return cmd
        # Fallback to POSIX grep.
        cmd = ["grep", "-rn", "--color=never"]
        if args.glob:
            cmd += [f"--include={args.glob}"]
        cmd += ["-e", args.pattern, target]
        return cmd

    async def call(self, args: GrepInput, context: ToolContext) -> ToolResult:
        cmd = self._build_command(args)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=context.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except OSError as exc:
            return ToolResult.fail(f"Search failed to start: {exc}")

        # Exit code 1 means "no matches" for both rg and grep — not an error.
        if proc.returncode not in (0, 1):
            return ToolResult.fail(
                stderr.decode(errors="replace").strip() or f"Search exited {proc.returncode}"
            )

        out = stdout.decode(errors="replace")
        if not out.strip():
            return ToolResult(output="No matches found.")
        if len(out) > MAX_OUTPUT_BYTES:
            spill = save_overflow(out, "Grep", context)
            return ToolResult(
                output=out[:MAX_OUTPUT_BYTES]
                + truncation_note(spill, shown=MAX_OUTPUT_BYTES, total=len(out))
            )
        return ToolResult(output=out.rstrip("\n"))
