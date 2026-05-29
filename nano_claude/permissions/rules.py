"""Allow/deny rule matching using fnmatch patterns.

A rule pattern is either a bare tool name (``"Read"``, ``"Bash"``) or a
tool-name plus an argument matcher (``"Bash(git *)"``). For the latter form the
argument pattern is matched against the tool's primary argument (``command`` for
Bash, ``file_path`` for file tools).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Literal

# Which argument each tool's parenthesised pattern matches against.
_PRIMARY_ARG = {
    "Bash": "command",
    "Read": "file_path",
    "Write": "file_path",
    "Edit": "file_path",
    "GlobTool": "pattern",
    "Grep": "pattern",
}


@dataclass(frozen=True)
class PermissionRule:
    pattern: str  # e.g. "Bash" or "Bash(git *)"
    decision: Literal["allow", "deny"]


def matches_rule(rule: PermissionRule, tool_name: str, args: dict[str, Any]) -> bool:
    if "(" not in rule.pattern:
        return fnmatch.fnmatch(tool_name, rule.pattern)

    # Split on the FIRST "(" and strip only the single matching trailing ")",
    # so the argument pattern may itself contain parentheses
    # (e.g. "Bash(git log --pretty=(short))").
    open_idx = rule.pattern.index("(")
    name_part = rule.pattern[:open_idx]
    arg_pattern = rule.pattern[open_idx + 1 :]
    if arg_pattern.endswith(")"):
        arg_pattern = arg_pattern[:-1]
    if not fnmatch.fnmatch(tool_name, name_part):
        return False
    arg_key = _PRIMARY_ARG.get(tool_name)
    value = str(args.get(arg_key, "")) if arg_key else ""
    return fnmatch.fnmatch(value, arg_pattern)


def first_match(
    rules: list[PermissionRule], tool_name: str, args: dict[str, Any]
) -> PermissionRule | None:
    return next((r for r in rules if matches_rule(r, tool_name, args)), None)
