"""Dataset adapter protocol — the seam for supporting new benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from evals.types import InstanceEval, Task


@runtime_checkable
class DatasetAdapter(Protocol):
    """Loads tasks and grades captured patches for one benchmark.

    Implementations live beside this file and register themselves in
    ``evals.datasets.REGISTRY``. The rollout phase is fully generic; only
    ``load`` (how to turn dataset rows into ``Task``s) and ``evaluate`` (how to
    grade) are dataset-specific.
    """

    name: str

    def load(self) -> list[Task]:
        """Return every task in the dataset, normalized to ``Task``."""
        ...

    def evaluate(
        self,
        predictions_path: Path,
        instance_ids: list[str],
        run_dir: Path,
        run_id: str,
        max_workers: int,
        timeout: int,
    ) -> dict[str, InstanceEval]:
        """Grade a predictions file, keyed by instance_id."""
        ...
