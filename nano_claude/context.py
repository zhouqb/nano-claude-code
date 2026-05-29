"""System prompt assembly.

Phase 1 keeps this minimal: identity + environment block (OS, shell, cwd, git,
date). Later phases extend it with tool guidance, CLAUDE.md, and memory.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import date
from pathlib import Path

IDENTITY = (
    "You are nano-claude-code, a minimal command-line coding assistant. "
    "You help the user with software-engineering tasks directly in their terminal. "
    "Be concise and direct. When you don't know something, say so."
)


def _git_branch(cwd: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch or None


def _is_git_repo(cwd: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def build_environment_block(cwd: str | None = None) -> str:
    cwd = cwd or os.getcwd()
    shell = os.environ.get("SHELL", "unknown")
    is_repo = _is_git_repo(cwd)
    lines = [
        "Here is information about the environment you are running in:",
        f"- Working directory: {cwd}",
        f"- Platform: {platform.system().lower()}",
        f"- OS version: {platform.platform()}",
        f"- Shell: {shell}",
        f"- Today's date: {date.today().isoformat()}",
        f"- Is a git repository: {'yes' if is_repo else 'no'}",
    ]
    if is_repo:
        branch = _git_branch(cwd)
        if branch:
            lines.append(f"- Current git branch: {branch}")
    return "\n".join(lines)


def build_system_prompt(cwd: str | None = None) -> str:
    cwd = cwd or os.getcwd()
    parts = [IDENTITY, build_environment_block(cwd)]
    claude_md = _read_project_instructions(cwd)
    if claude_md:
        parts.append(
            "The following project instructions come from CLAUDE.md and must be "
            "followed:\n\n" + claude_md
        )
    return "\n\n".join(parts)


def _read_project_instructions(cwd: str) -> str | None:
    path = Path(cwd) / "CLAUDE.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
