"""CLI entry point for the eval pipeline: ``python -m evals.run``.

Everything runs inside the official SWE-bench Docker images. For each instance,
nano-claude rolls out *inside* the instance container (exact interpreter +
prebuilt deps, can run the project's tests), then the patch is graded with the
same image. Images are pre-built once up front; rollout + grade run on a flat
pool of workers (no repo affinity is needed — each task has its own container),
and grades run concurrently, each under its own harness run_id.

Outputs (in the run dir):
- ``analysis.csv`` — one row per instance (the results table).
- ``predictions.jsonl`` / ``results.jsonl`` / ``summary.json`` — standard records.
- with ``--failure-study`` (default): ``failures/<id>/`` bundles each miss's
  agent log, patch, report, and test output for offline review.

    python -m evals.run --dataset swe-bench-verified --sample 50 --workers 5
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
from rich.console import Console

from evals.config import RolloutConfig
from evals.datasets import available, get_adapter
from evals.docker_rollout import (
    prebuild_images,
    prepare_tooling,
    run_task_docker,
    warm_tooling,
)
from evals.report import aggregate, build_results, print_summary, write_results
from evals.types import EvalStatus, InstanceEval, RolloutStatus

console = Console()

MAX_WORKERS = 5

CSV_FIELDS = [
    "idx",
    "instance_id",
    "repo",
    "rollout_status",
    "eval_status",
    "resolved",
    "failure_category",
    "f2p_passed",
    "f2p_total",
    "p2p_failed",
    "rollout_s",
    "eval_s",
    "patch_bytes",
    "env_ready",
    "error",
    "patch_file",
    "agent_log",
    "test_output",
]


def _instance_log_dir(run_dir: Path, run_id: str, model: str, iid: str) -> Path:
    return Path(run_dir) / "logs" / "run_evaluation" / run_id / model.replace("/", "__") / iid


def _load_report(run_dir: Path, run_id: str, model: str, iid: str) -> dict | None:
    path = _instance_log_dir(run_dir, run_id, model, iid) / "report.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())[iid]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _classify(r, verdict: InstanceEval, report: dict | None) -> str:
    """A coarse failure bucket for an unresolved instance."""
    if r.status is RolloutStatus.TIMEOUT:
        return "agent_timeout"
    if r.status is RolloutStatus.ERROR:
        return "agent_error"
    if r.status is RolloutStatus.EMPTY_PATCH or verdict.status is EvalStatus.EMPTY_PATCH:
        return "empty_patch"
    if verdict.status is EvalStatus.SKIPPED:
        return "not_evaluated"
    if report is None:
        return "grading_error"
    if not report.get("patch_successfully_applied", True):
        return "patch_apply_failed"
    return "tests_failed"


def _test_counts(report: dict | None) -> tuple[int, int, int]:
    """(FAIL_TO_PASS passed, FAIL_TO_PASS total, PASS_TO_PASS failed)."""
    if not report:
        return 0, 0, 0
    ts = report.get("tests_status", {})
    f2p = ts.get("FAIL_TO_PASS", {})
    p2p = ts.get("PASS_TO_PASS", {})
    passed = len(f2p.get("success", []))
    return passed, passed + len(f2p.get("failure", [])), len(p2p.get("failure", []))


def _bundle_failure(dest: Path, r, src_log: Path) -> None:
    """Copy a failure's artifacts into one folder for easy review."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model_patch.diff").write_text(r.model_patch)
    if r.log_path and Path(r.log_path).is_file():
        shutil.copy(r.log_path, dest / "agent.log")
    for name in ("report.json", "test_output.txt", "run_instance.log"):
        if (src_log / name).is_file():
            shutil.copy(src_log / name, dest / name)


def _done_ids(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    with csv_path.open() as f:
        return {row["instance_id"] for row in csv.DictReader(f)}


def _select(
    tasks: list, instance_ids: str, repos: str, sample: int, offset: int, seed: int
) -> list:
    """Pick the instances to run.

    Explicit ``--instance-ids`` win outright. Otherwise we deterministically
    order the dataset by ``seed``, optionally filter by repo, then take the
    window ``[offset : offset+sample]`` (sample 0 = to the end). Running
    --offset 0/100/200 ... with one seed gives non-overlapping batches.
    """
    if instance_ids.strip():
        wanted = {s.strip() for s in instance_ids.split(",") if s.strip()}
        return [t for t in tasks if t.instance_id in wanted]
    ordered = random.Random(seed).sample(tasks, len(tasks))
    if repos.strip():
        wanted_repos = {s.strip() for s in repos.split(",") if s.strip()}
        ordered = [t for t in ordered if t.repo in wanted_repos]
    end = offset + sample if sample else None
    return ordered[offset:end]


@click.command()
@click.option(
    "--dataset",
    default="swe-bench-lite",
    show_default=True,
    help=f"Dataset to run. Available: {', '.join(available())}.",
)
@click.option("--split", default="test", show_default=True)
@click.option("--sample", default=0, show_default=True, help="Window size (0 = whole dataset).")
@click.option(
    "--offset",
    default=0,
    show_default=True,
    help="Skip the first N of the (seeded) ordering. Run --offset 0/100/200 ... "
    "with the same --seed for non-overlapping batches.",
)
@click.option("--seed", default=0, show_default=True, help="Sampling seed (reproducible).")
@click.option("--instance-ids", default="", help="Comma-separated ids to run instead of sampling.")
@click.option("--repos", default="", help="Comma-separated repos (owner/name) to restrict to.")
@click.option("--output", default=None, help="Run directory (default: runs/<dataset>-<n>-<seed>).")
@click.option("--model", default=None, help="Agent model (default: nano-claude's default).")
@click.option("--max-turns", default=200, show_default=True)
@click.option("--task-timeout", default=1800, show_default=True, help="Per-case agent budget (s).")
@click.option("--eval-timeout", default=1800, show_default=True, help="Per-case test timeout (s).")
@click.option(
    "--workers",
    default=MAX_WORKERS,
    show_default=True,
    help=f"Concurrent workers (flat pool, capped at {MAX_WORKERS}).",
)
@click.option(
    "--grade-workers",
    default=0,
    show_default=True,
    help="Concurrent Docker grades (0 = match --workers). Live containers stay "
    "bounded by --workers since each worker rolls out then grades in turn.",
)
@click.option(
    "--build-workers",
    default=8,
    show_default=True,
    help="Concurrent image builds in the one-time prebuild phase. Decoupled from "
    "--workers since building distinct images is independent and CPU/IO-bound; "
    "keep it modest (heavy docker builds can thrash).",
)
@click.option(
    "--eval/--no-eval",
    "do_eval",
    default=True,
    show_default=True,
    help="Grade each patch with the Docker harness (off = rollout only).",
)
@click.option(
    "--failure-study/--no-failure-study",
    default=True,
    show_default=True,
    help="Bundle each failure's artifacts under failures/<id>/ for review.",
)
@click.option(
    "--no-strip-test-changes",
    is_flag=True,
    help="Keep agent edits to test files in the captured patch.",
)
@click.option(
    "--nano-bin", default="", help="Path to the nano-claude binary (default: autodetect)."
)
@click.option("--resume", is_flag=True, help="Skip cases already in analysis.csv.")
def main(
    dataset,
    split,
    sample,
    offset,
    seed,
    instance_ids,
    repos,
    output,
    model,
    max_turns,
    task_timeout,
    eval_timeout,
    workers,
    grade_workers,
    build_workers,
    do_eval,
    failure_study,
    no_strip_test_changes,
    nano_bin,
    resume,
):
    """Evaluate nano-claude-code on a SWE-bench dataset, inside Docker."""
    try:
        adapter = get_adapter(dataset)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from None
    adapter.split = split  # the swe-bench adapter reads self.split in load()/evaluate()

    default_name = f"{dataset}-{sample or 'all'}-{seed}" + (f"-off{offset}" if offset else "")
    run_dir = Path(output) if output else Path("runs") / default_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    csv_path = run_dir / "analysis.csv"
    preds_dir = run_dir / "predictions"
    patches_dir = run_dir / "patches"
    rollout_logs = run_dir / "logs"
    for d in (preds_dir, patches_dir):
        d.mkdir(exist_ok=True)

    cfg = RolloutConfig(
        max_turns=max_turns,
        task_timeout=task_timeout,
        nano_bin=nano_bin,
        strip_test_changes=not no_strip_test_changes,
    )
    if model:
        cfg.model = model

    console.print(f"[bold]Loading[/bold] {dataset}/{split}...")
    sampled = _select(adapter.load(), instance_ids, repos, sample, offset, seed)
    if not sampled:
        raise click.ClickException("No tasks selected.")

    done = _done_ids(csv_path) if resume else set()
    pending = [t for t in sampled if t.instance_id not in done]
    idx_of = {t.instance_id: i for i, t in enumerate(sampled, start=1)}
    total = len(sampled)

    # Build every image up front (no build races during rollout) + the shared
    # nano-claude tooling. Drop instances whose image failed to build.
    tooling = None
    if pending:
        console.print(
            f"[dim]docker: pre-building images for {len(pending)} instance(s) "
            f"({build_workers} build workers)…[/dim]"
        )
        built = prebuild_images(
            [t.extra["instance"] for t in pending], build_workers, run_dir / "image-build-logs"
        )
        skipped = [t for t in pending if t.instance_id not in built]
        if skipped:
            console.print(
                f"[yellow]docker: {len(skipped)} image build(s) failed; skipping those.[/yellow]"
            )
        pending = [t for t in pending if t.instance_id in built]
        tooling = prepare_tooling(run_dir / "tooling")
        if pending:
            # Populate the shared uv cache/interpreter once so concurrent per-task
            # setups are cache reads (no download race).
            console.print("[dim]docker: warming shared uv cache (one-time)…[/dim]")
            warm_tooling(pending[0], tooling)

    n_workers = max(1, min(workers, MAX_WORKERS, len(pending))) if pending else 0
    n_grade = grade_workers if grade_workers > 0 else n_workers
    n_grade = max(1, min(n_grade, n_workers)) if n_workers else 0

    console.print(
        f"[bold]{total} task(s)[/bold] from {dataset}/{split} "
        f"(seed={seed}, offset={offset}). Agent={cfg.model}. "
        f"workers={n_workers}, grade-workers={n_grade}, eval={do_eval}.\n"
        f"Output: {run_dir}/  | resume-skipping {len(done)}, {len(pending)} to run."
    )

    grade_sem = threading.BoundedSemaphore(n_grade) if n_grade else None
    io_lock = threading.Lock()
    rollouts: list = []
    evals: dict[str, InstanceEval] = {}

    new_file = not csv_path.is_file()
    csv_file = csv_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        csv_file.flush()

    def run_one(task) -> None:
        iid = task.instance_id
        grade_run_id = f"{run_id}-{iid}"  # isolate concurrent harness processes
        # --- rollout (agent runs inside the instance container) ---
        r = run_task_docker(task, cfg, rollout_logs, run_id, tooling)
        patch_file = patches_dir / f"{iid}.diff"
        patch_file.write_text(r.model_patch)

        # --- grade (Docker harness); concurrent up to grade_sem ---
        eval_s = 0.0
        verdict = InstanceEval(EvalStatus.SKIPPED)
        if do_eval and r.model_patch.strip():
            verdict = InstanceEval(EvalStatus.EMPTY_PATCH)  # overwritten on success
            preds_path = preds_dir / f"{iid}.jsonl"
            preds_path.write_text(
                json.dumps(
                    {
                        "instance_id": iid,
                        "model_name_or_path": cfg.model_name_or_path,
                        "model_patch": r.model_patch,
                    }
                )
                + "\n"
            )
            t0 = time.monotonic()
            with grade_sem:
                try:
                    verdict = adapter.evaluate(
                        preds_path, [iid], run_dir, grade_run_id, 1, eval_timeout
                    ).get(iid, InstanceEval(EvalStatus.ERROR))
                except Exception as exc:  # noqa: BLE001 - a Docker hiccup must not stop the run
                    console.print(f"[red]grade error {iid}: {exc}[/red]")
                    verdict = InstanceEval(EvalStatus.ERROR)
            eval_s = time.monotonic() - t0

        report = _load_report(run_dir, grade_run_id, cfg.model_name_or_path, iid)
        resolved = verdict.resolved
        f2p_pass, f2p_total, p2p_fail = _test_counts(report)
        inst_log = _instance_log_dir(run_dir, grade_run_id, cfg.model_name_or_path, iid)
        test_output = inst_log / "test_output.txt"

        category = ""
        if not resolved:
            category = _classify(r, verdict, report)
            if failure_study:
                _bundle_failure(run_dir / "failures" / iid, r, inst_log)

        row = {
            "idx": idx_of[iid],
            "instance_id": iid,
            "repo": task.repo,
            "rollout_status": r.status.value,
            "eval_status": verdict.status.value,
            "resolved": resolved,
            "failure_category": category,
            "f2p_passed": f2p_pass,
            "f2p_total": f2p_total,
            "p2p_failed": p2p_fail,
            "rollout_s": round(r.duration_s, 1),
            "eval_s": round(eval_s, 1),
            "patch_bytes": len(r.model_patch),
            "env_ready": "" if r.env_ready is None else r.env_ready,
            "error": (r.error or "")[:300],
            "patch_file": str(patch_file),
            "agent_log": r.log_path or "",
            "test_output": str(test_output) if test_output.is_file() else "",
        }
        mark = "PASS" if resolved else (f"FAIL/{category}" if category else "done")
        err = f" :: {r.error}" if (not resolved and r.error) else ""
        line = (
            f"[{idx_of[iid]}/{total}] {iid}: {mark} "
            f"(rollout {r.duration_s:.0f}s, eval {eval_s:.0f}s){err}"
        )
        with io_lock:
            writer.writerow(row)
            csv_file.flush()
            rollouts.append(r)
            evals[iid] = verdict
            console.print(("[green]" if resolved else "[yellow]") + line + "[/]")

    if pending:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fut in [pool.submit(run_one, t) for t in pending]:
                fut.result()  # surface any worker-thread exception

    csv_file.close()

    # Combined predictions + standard results/summary.
    (run_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": r.instance_id,
                    "model_name_or_path": cfg.model_name_or_path,
                    "model_patch": r.model_patch,
                }
            )
            + "\n"
            for r in rollouts
        )
    )
    results = build_results(rollouts, evals if do_eval else {})
    summary = aggregate(results)
    write_results(run_dir, results, summary)
    print_summary(console, summary, evaluated=do_eval)
    console.print(f"\n[dim]Full results in {run_dir}/[/dim]")


if __name__ == "__main__":
    main()
