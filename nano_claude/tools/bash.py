"""Bash tool: run a shell command with a timeout."""

from __future__ import annotations

import asyncio
import re

from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult

DEFAULT_TIMEOUT_S = 120
MAX_OUTPUT_BYTES = 60_000

# Patterns that are almost never intentional and would be catastrophic.
DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf][a-zA-Z]*\s+(-{1,2}\S+\s+)*/(\s|$)"),
    re.compile(r"\brm\s+-rf\s+/\b"),
    re.compile(r":\(\)\s*\{.*\};\s*:"),  # fork bomb
    re.compile(r"\bmkfs\.\w+\s+/dev/"),
    re.compile(r"\bdd\b.*\bof=/dev/(sd|nvme|disk)"),
    re.compile(r">\s*/dev/(sd|nvme|disk)"),
]


def is_dangerous(command: str) -> bool:
    return any(pat.search(command) for pat in DANGEROUS_PATTERNS)


class BashInput(BaseModel):
    command: str = Field(description="The shell command to run.")
    timeout: int = Field(
        default=DEFAULT_TIMEOUT_S, description="Timeout in seconds before the command is killed."
    )


class BashTool(Tool):
    name = "Bash"
    description = (
        "Run a shell command and return its combined stdout/stderr. Commands run "
        "in the current working directory with a timeout."
    )
    input_schema = BashInput

    async def check_permissions(self, args: BashInput, context: ToolContext) -> PermissionDecision:
        if is_dangerous(args.command):
            return PermissionDecision(
                behavior="deny",
                reason="Command matches a blocked destructive pattern.",
            )
        return PermissionDecision(behavior="ask", prompt=f"Run: {args.command}")

    async def call(self, args: BashInput, context: ToolContext) -> ToolResult:
        # Defense in depth: check_permissions already denies dangerous commands
        # (and a deny can't be overridden by any permission mode), so the manager
        # never reaches here for one. This guard is a hard safety floor for any
        # caller that might invoke call() without going through the manager.
        if is_dangerous(args.command):
            return ToolResult.fail("Refused: command matches a blocked destructive pattern.")

        try:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                cwd=context.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return ToolResult.fail(f"Failed to start command: {exc}")

        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=args.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult.fail(f"Command timed out after {args.timeout}s.")

        out = stdout.decode(errors="replace")
        if len(out) > MAX_OUTPUT_BYTES:
            # TODO: spill the full output to a temp file (e.g. under the session
            # dir) and reference its path in the truncated result, so large
            # outputs aren't lost permanently. Factor this into a shared helper
            # reused by Grep/Glob (see their matching TODOs).
            out = out[:MAX_OUTPUT_BYTES] + "\n... (output truncated)"

        if proc.returncode != 0:
            return ToolResult(
                output=f"(exit code {proc.returncode})\n{out}".rstrip(),
                is_error=True,
                error=f"Command exited with code {proc.returncode}",
            )
        return ToolResult(output=out.rstrip() or "(no output)")
