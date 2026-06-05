"""SWE-bench Lite adapter: HF dataset loading + official Docker grading."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from evals.prompts import swe_bench_prompt
from evals.types import EvalStatus, InstanceEval, Task


class SweBenchLiteAdapter:
    """Loads SWE-bench Lite and grades via ``swebench.harness.run_evaluation``."""

    name = "swe-bench-lite"
    dataset_name = "SWE-bench/SWE-bench_Lite"
    split = "test"

    def load(self) -> list[Task]:
        # Imported lazily so the eval extra is only needed when actually used.
        from swebench.harness.utils import load_swebench_dataset

        rows = load_swebench_dataset(self.dataset_name, self.split)
        tasks: list[Task] = []
        for row in rows:
            tasks.append(
                Task(
                    instance_id=row["instance_id"],
                    repo=row["repo"],
                    base_commit=row["base_commit"],
                    prompt=swe_bench_prompt(row["repo"], row["problem_statement"]),
                    extra={"test_patch": row.get("test_patch", "")},
                )
            )
        return tasks

    def evaluate(
        self,
        predictions_path: Path,
        instance_ids: list[str],
        run_dir: Path,
        run_id: str,
        max_workers: int,
        timeout: int,
    ) -> dict[str, InstanceEval]:
        """Shell out to the official harness, then read its per-instance reports.

        The harness builds/pulls per-instance Docker images, applies the model
        patch + gold test patch, runs FAIL_TO_PASS / PASS_TO_PASS, and writes a
        ``report.json`` per instance under ``logs/run_evaluation/<run_id>/...``.
        We run it with ``cwd=run_dir`` so all of its artifacts (logs + the
        ``<model>.<run_id>.json`` summary) stay contained in the run directory,
        then read the per-instance reports — the location-stable source of truth
        (the top-level summary path varies between harness versions).
        """
        predictions_path = Path(predictions_path).resolve()
        run_dir = Path(run_dir)
        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--split",
            self.split,
            "--predictions_path",
            str(predictions_path),
            "--run_id",
            run_id,
            "--max_workers",
            str(max_workers),
            "--timeout",
            str(timeout),
            "--instance_ids",
            *instance_ids,
        ]
        # Stream the harness output through; it's long-running and informative.
        subprocess.run(cmd, check=True, cwd=run_dir)
        model = self._model_name(predictions_path)
        return self._parse_reports(run_dir, run_id, model, instance_ids)

    @staticmethod
    def _model_name(predictions_path: Path) -> str:
        """The ``model_name_or_path`` the predictions were written under."""
        with predictions_path.open() as f:
            for line in f:
                if line.strip():
                    return json.loads(line)["model_name_or_path"]
        return "model"

    def _parse_reports(
        self, run_dir: Path, run_id: str, model: str, instance_ids: list[str]
    ) -> dict[str, InstanceEval]:
        from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR

        base = Path(run_dir) / RUN_EVALUATION_LOG_DIR / run_id / model.replace("/", "__")
        out: dict[str, InstanceEval] = {}
        for iid in instance_ids:
            report = base / iid / LOG_REPORT
            if not report.is_file():
                # No report written — the harness errored on this instance.
                out[iid] = InstanceEval(EvalStatus.ERROR)
                continue
            try:
                verdict = json.loads(report.read_text())[iid]
                resolved = bool(verdict.get("resolved"))
            except (json.JSONDecodeError, KeyError, OSError):
                out[iid] = InstanceEval(EvalStatus.ERROR)
                continue
            out[iid] = InstanceEval(
                EvalStatus.RESOLVED if resolved else EvalStatus.UNRESOLVED,
                resolved=resolved,
                detail=verdict.get("tests_status", {}),
            )
        return out
