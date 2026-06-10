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
    max_turns: int = 500
    # Reasoning/thinking effort passed through to nano-claude (and on to litellm).
    # None ⇒ omit the flag (provider default); see AgentConfig.reasoning_effort.
    reasoning_effort: str | None = os.environ.get("NANO_CLAUDE_REASONING_EFFORT") or None
    # Per-task wall-clock budget for the agent process (seconds).
    task_timeout: int = 3600
    nano_bin: str = ""
    # Drop any edits the agent made to graded test files before capturing the
    # patch (they would collide with the harness's gold test patch).
    strip_test_changes: bool = True
    # Run a second, independent nano-claude pass after the first one lands a
    # non-empty patch: an adversarial verifier that re-checks the fix against
    # the issue, hunts regressions in the existing suite, and repairs what it
    # finds. Off by default; opt in with NANO_CLAUDE_VERIFY_PASS=1. See
    # docker_rollout._run_verification_pass and prompts.verification_pass_prompt.
    verify_pass: bool = bool(os.environ.get("NANO_CLAUDE_VERIFY_PASS"))
    # Wall-clock budget for the verification pass (seconds); shorter than the
    # main rollout since it is checking/repairing an existing fix, not solving
    # from scratch — but still generous enough for max-reasoning to run tests.
    verify_timeout: int = 1800

    def resolved_bin(self) -> str:
        return self.nano_bin or default_nano_bin()

    @property
    def model_name_or_path(self) -> str:
        """Identifier embedded in predictions + harness report filenames."""
        return "nano-claude__" + self.model.replace("/", "-")
