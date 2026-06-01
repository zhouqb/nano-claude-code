"""Bash tool: run a shell command with a timeout."""

from __future__ import annotations

import asyncio
import re
import shlex

from pydantic import BaseModel, Field

from nano_claude.tools.base import PermissionDecision, Tool, ToolContext, ToolResult
from nano_claude.tools.overflow import save_overflow, truncation_note

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


# Commands that only read or inspect — safe for the read-only carve-out used by
# memory extraction. Anything not on this allowlist is treated as write-capable.
_READ_ONLY_COMMANDS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "find",
        "pwd",
        "echo",
        "printf",
        "stat",
        "file",
        "tree",
        "which",
        "type",
        "whoami",
        "id",
        "date",
        "env",
        "du",
        "df",
        "dirname",
        "basename",
        "realpath",
        "readlink",
        "sort",
        "uniq",
        "cut",
        "comm",
        "diff",
        "cmp",
        "tac",
        "nl",
        "column",
        "hexdump",
        "xxd",
        "sha256sum",
        "md5sum",
        "true",
    }
)
# git subcommands that never mutate the repo (write-capable ones like branch,
# tag, remote, config, push are intentionally excluded).
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "ls-files",
        "rev-parse",
        "blame",
        "describe",
        "cat-file",
        "shortlog",
        "ls-tree",
        "reflog",
        "whatchanged",
    }
)


def is_read_only(command: str) -> bool:
    """Best-effort: True only if every part of ``command`` merely reads/inspects.

    Conservative by design — when in doubt it returns False. Rejects output
    redirection and command substitution outright, then requires every segment of
    a pipeline/sequence to be an allow-listed read-only command (or a read-only
    ``git`` subcommand). Used by the memory-extraction permission gate.
    """
    if ">" in command or "$(" in command or "`" in command:
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False

    segment: list[str] = []
    segments: list[list[str]] = []
    for tok in tokens:
        if tok in ("|", "||", "&&", ";", "&"):
            segments.append(segment)
            segment = []
        else:
            segment.append(tok)
    segments.append(segment)

    for seg in segments:
        if not seg:
            continue
        cmd = seg[0]
        if cmd == "git":
            sub = next((t for t in seg[1:] if not t.startswith("-")), "")
            if sub not in _READ_ONLY_GIT_SUBCOMMANDS:
                return False
        elif cmd not in _READ_ONLY_COMMANDS:
            return False
    return True


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
            spill = save_overflow(out, "Bash", context)
            out = out[:MAX_OUTPUT_BYTES] + truncation_note(
                spill, shown=MAX_OUTPUT_BYTES, total=len(out)
            )

        if proc.returncode != 0:
            return ToolResult(
                output=f"(exit code {proc.returncode})\n{out}".rstrip(),
                is_error=True,
                error=f"Command exited with code {proc.returncode}",
            )
        return ToolResult(output=out.rstrip() or "(no output)")
