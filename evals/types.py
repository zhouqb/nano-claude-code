"""Shared dataclasses and enums for the eval pipeline.

These are dataset-agnostic on purpose: a ``Task`` carries only what the rollout
phase needs (repo, commit, prompt), while the opaque ``extra`` mapping holds
whatever a dataset's grader requires (e.g. SWE-bench's ``test_patch``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RolloutStatus(StrEnum):
    """Outcome of running the agent on a task (independent of grading)."""

    OK = "ok"  # agent finished and produced a non-empty patch
    EMPTY_PATCH = "empty_patch"  # agent finished but changed nothing
    ERROR = "error"  # agent process failed (non-zero exit / crash)
    TIMEOUT = "timeout"  # agent exceeded the per-task wall-clock budget


class EvalStatus(StrEnum):
    """Outcome of grading a rollout."""

    RESOLVED = "resolved"  # all gold tests pass
    UNRESOLVED = "unresolved"  # graded, but tests did not pass
    EMPTY_PATCH = "empty_patch"  # nothing to grade
    ERROR = "error"  # grader failed (build/apply/harness error)
    SKIPPED = "skipped"  # evaluation phase not run


@dataclass(frozen=True)
class Task:
    """A single benchmark instance, normalized across datasets."""

    instance_id: str
    repo: str  # "owner/name" on GitHub
    base_commit: str
    prompt: str
    # Dataset-specific grading payload (kept opaque to the rollout phase).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """What the rollout phase records for one task."""

    instance_id: str
    repo: str
    base_commit: str
    status: RolloutStatus
    model_patch: str = ""
    duration_s: float = 0.0
    error: str | None = None
    log_path: str | None = None


@dataclass
class InstanceEval:
    """Grading verdict for one instance."""

    status: EvalStatus
    resolved: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """The fully-joined per-task record written to ``results.jsonl``."""

    instance_id: str
    repo: str
    rollout_status: RolloutStatus
    eval_status: EvalStatus
    resolved: bool
    duration_s: float
    patch_size: int
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "rollout_status": self.rollout_status.value,
            "eval_status": self.eval_status.value,
            "resolved": self.resolved,
            "duration_s": round(self.duration_s, 2),
            "patch_size": self.patch_size,
            "error": self.error,
        }
