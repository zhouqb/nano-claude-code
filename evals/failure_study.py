"""Sequential data-collection run over a sample of SWE-bench test cases.

This is deliberately analysis-free: for each sampled instance, one at a time, it
runs nano-claude (rollout), grades it with the Docker harness, and persists the
exact graded git diff, the agent transcript, and the harness test output — plus
a CSV row with the factual verdict. The ``improvement_suggestion`` column is left
blank on purpose: failure analysis is done afterwards by a human/strong model
reading the saved artifacts, not by an automated small-model call during the run.

Run:
    python -m evals.failure_study --sample 50 --seed 0 --output runs/study50
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
from evals.datasets import get_adapter
from evals.docker_rollout import (
    prebuild_images,
    prepare_tooling,
    run_task_docker,
    warm_tooling,
)
from evals.repo_cache import RepoCache
from evals.rollout import run_task
from evals.scheduler import MAX_WORKERS, partition_by_repo
from evals.types import EvalStatus, InstanceEval, RolloutStatus
from evals.venv_env import filter_feasible, make_env_provider

console = Console()

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
    "improvement_suggestion",  # left blank — filled in during morning analysis
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
    if r.status is RolloutStatus.TIMEOUT:
        return "agent_timeout"
    if r.status is RolloutStatus.ERROR:
        return "agent_error"
    if r.status is RolloutStatus.EMPTY_PATCH or verdict.status is EvalStatus.EMPTY_PATCH:
        return "empty_patch"
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
    """Copy a failure's artifacts into one folder for easy morning review."""
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


@click.command()
@click.option("--dataset", default="swe-bench-lite", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--sample", default=50, show_default=True, help="Number of cases to sample.")
@click.option("--seed", default=0, show_default=True, help="Sampling seed (reproducible).")
@click.option("--instance-ids", default="", help="Comma-separated ids to run instead of sampling.")
@click.option("--output", default=None, help="Run directory (default: runs/study-<n>-<seed>).")
@click.option("--model", default=None, help="Agent model (default: nano-claude's default).")
@click.option("--max-turns", default=200, show_default=True)
@click.option("--task-timeout", default=1800, show_default=True, help="Per-case agent budget (s).")
@click.option("--eval-timeout", default=1800, show_default=True, help="Per-case test timeout (s).")
@click.option("--repo-root", default="/tmp/nano-swebench/repos", show_default=True)
@click.option(
    "--rollout-backend",
    type=click.Choice(["host", "host-venv", "docker"]),
    default="host",
    show_default=True,
    help="host: bare clone (no test env). host-venv: per-(repo,version) venv so the "
    "agent can run tests. docker: run the agent inside the SWE-bench instance "
    "container (covers all instances, incl. compiled-dep ones).",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    help=f"Concurrent rollout workers (repo-affinity; capped at {MAX_WORKERS}).",
)
@click.option(
    "--grade-workers",
    default=0,
    show_default=True,
    help="Concurrent Docker grades (0 = match --workers). Total live containers is "
    "still bounded by --workers since each worker rolls out then grades in turn.",
)
@click.option("--resume", is_flag=True, help="Skip cases already in the CSV.")
def main(
    dataset,
    split,
    sample,
    seed,
    instance_ids,
    output,
    model,
    max_turns,
    task_timeout,
    eval_timeout,
    repo_root,
    rollout_backend,
    workers,
    grade_workers,
    resume,
):
    """Collect rollout + grade artifacts for a sample of cases (no auto-analysis)."""
    adapter = get_adapter(dataset)
    adapter.split = split  # the swe-bench adapter reads self.split in load()/evaluate()

    run_dir = Path(output) if output else Path("runs") / f"study-{sample}-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    csv_path = run_dir / "analysis.csv"
    progress = run_dir / "progress.log"
    preds_dir = run_dir / "predictions"
    patches_dir = run_dir / "patches"
    rollout_logs = run_dir / "logs"
    for d in (preds_dir, patches_dir):
        d.mkdir(exist_ok=True)

    cfg = RolloutConfig(max_turns=max_turns, task_timeout=task_timeout)
    if model:
        cfg.model = model

    tasks = adapter.load()
    # host-venv only supports the feasible (Python>=3.8, no compiled deps) slice;
    # restrict the pool before sampling so --sample N gives N runnable cases.
    env_provider = make_env_provider(rollout_backend, Path(repo_root).parent / "venvs")
    if rollout_backend == "host-venv":
        pool = filter_feasible(tasks)
        console.print(
            f"[dim]host-venv: {len(pool)}/{len(tasks)} instances feasible (rest skipped).[/dim]"
        )
        tasks = pool
    if instance_ids.strip():
        wanted = {s.strip() for s in instance_ids.split(",") if s.strip()}
        sampled = [t for t in tasks if t.instance_id in wanted]
    else:
        sampled = random.Random(seed).sample(tasks, min(sample, len(tasks)))
    # Group same-repo cases together so the clone and the Docker env image are
    # reused across them (the biggest time sink). Still strictly one at a time.
    sampled.sort(key=lambda t: t.repo)

    done = _done_ids(csv_path) if resume else set()
    cache = RepoCache(Path(repo_root))

    new_file = not csv_path.is_file()
    csv_file = csv_path.open("a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        csv_file.flush()

    pending = [t for t in sampled if t.instance_id not in done]
    idx_of = {t.instance_id: i for i, t in enumerate(sampled, start=1)}
    total = len(sampled)

    # Docker backend: build all images up front (no build races during rollout)
    # and build the shared nano-claude tooling. Drop instances whose image
    # failed to build so workers don't trip over a missing image.
    tooling = None
    if rollout_backend == "docker" and pending:
        console.print(f"[dim]docker: pre-building images for {len(pending)} instance(s)…[/dim]")
        built = prebuild_images(
            [t.extra["instance"] for t in pending], workers, run_dir / "image-build-logs"
        )
        skipped = [t for t in pending if t.instance_id not in built]
        if skipped:
            console.print(
                f"[yellow]docker: {len(skipped)} image build(s) failed; skipping those.[/yellow]"
            )
        pending = [t for t in pending if t.instance_id in built]
        tooling = prepare_tooling(run_dir / "tooling")
        if pending:
            # Populate the shared uv cache/interpreter once before the pool, so
            # concurrent per-task setups are cache reads (no download race).
            console.print("[dim]docker: warming shared uv cache (one-time)…[/dim]")
            warm_tooling(pending[0], tooling)

    # Repo-affinity buckets (same repo -> one worker, so its clone/venv is never
    # touched concurrently); caps at MAX_WORKERS and at the number of repos.
    buckets = partition_by_repo(pending, workers)
    n_workers = len(buckets)

    # Grades run concurrently up to n_grade. Total live containers stays <= n_workers
    # (each worker thread rolls out *then* grades, never both at once), so this only
    # lifts the redundant serial-grade throttle -- it doesn't raise the memory ceiling.
    n_grade = grade_workers if grade_workers > 0 else n_workers
    n_grade = max(1, min(n_grade, n_workers))

    console.print(
        f"[bold]Study (collect-only)[/bold]: {len(sampled)} case(s) from "
        f"{dataset}/{split} (seed={seed}). Agent={cfg.model}. "
        f"workers={n_workers}, grade-workers={n_grade}.\n"
        f"Output: {run_dir}/  | resume-skipping {len(done)}, {len(pending)} to run."
    )

    grade_sem = threading.BoundedSemaphore(n_grade)  # cap concurrent Docker grades
    io_lock = threading.Lock()  # serialize CSV/console/progress + counters
    stats = {"ran": 0, "resolved": 0}

    def run_one(task) -> None:
        iid = task.instance_id
        # Per-grade run_id isolates concurrent harness processes (container names,
        # log dirs, summary file). Used for grading + report lookup below.
        grade_run_id = f"{run_id}-{iid}"
        # --- rollout (agent); parallel across workers ---
        if rollout_backend == "docker":
            r = run_task_docker(task, cfg, rollout_logs, run_id, tooling)
        else:
            r = run_task(task, cache, cfg, rollout_logs, env_provider=env_provider)
        patch_file = patches_dir / f"{iid}.diff"
        patch_file.write_text(r.model_patch)

        # --- grade (Docker harness); concurrent up to grade_sem ---
        eval_s = 0.0
        verdict = InstanceEval(EvalStatus.EMPTY_PATCH)
        if r.model_patch.strip():
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
            "improvement_suggestion": "",
        }
        mark = "PASS" if resolved else f"FAIL/{category}"
        err = f" :: {r.error}" if (not resolved and r.error) else ""
        line = (
            f"[{idx_of[iid]}/{total}] {iid}: {mark} "
            f"(rollout {r.duration_s:.0f}s, eval {eval_s:.0f}s){err}"
        )
        with io_lock:
            writer.writerow(row)
            csv_file.flush()
            console.print(("[green]" if resolved else "[yellow]") + line + "[/]")
            with progress.open("a") as p:
                p.write(line + "\n")
            stats["ran"] += 1
            stats["resolved"] += int(resolved)

    def worker(bucket) -> None:
        for task in bucket:
            run_one(task)

    if buckets:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fut in [pool.submit(worker, b) for b in buckets]:
                fut.result()  # surface any worker-thread exception

    csv_file.close()
    console.print(
        f"\n[bold]Done.[/bold] Ran {stats['ran']} case(s); resolved {stats['resolved']}. "
        f"CSV: {csv_path}  | failures bundled under {run_dir}/failures/"
    )


if __name__ == "__main__":
    main()
