"""Run-wide configuration for the rollout phase."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


def default_nano_bin() -> str:
    """Locate the nano-claude CLI: prefer the project venv, then PATH."""
    venv = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "nano-claude"
    if venv.exists():
        return str(venv)
    found = shutil.which("nano-claude")
    return found or "nano-claude"


@dataclass
class RolloutConfig:
    """Knobs for invoking the agent on each task."""

    model: str = os.environ.get("NANO_CLAUDE_MODEL", "deepseek/deepseek-v4-flash")
    max_turns: int = 200
    # Per-task wall-clock budget for the agent process (seconds).
    task_timeout: int = 1800
    nano_bin: str = ""
    # Drop any edits the agent made to graded test files before capturing the
    # patch (they would collide with the harness's gold test patch).
    strip_test_changes: bool = True

    def resolved_bin(self) -> str:
        return self.nano_bin or default_nano_bin()

    @property
    def model_name_or_path(self) -> str:
        """Identifier embedded in predictions + harness report filenames."""
        return "nano-claude__" + self.model.replace("/", "-")
