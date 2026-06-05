"""CLI entry point for the eval pipeline: ``python -m evals.run``.

Phase A (rollout) runs the agent across repo-partitioned workers and writes
``predictions.jsonl`` + ``rollout_records.jsonl``. Phase B (eval) grades the
predictions via the dataset's adapter. Phase C aggregates into ``results.jsonl``
+ ``summary.json`` and prints a summary table.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console

from evals.config import RolloutConfig
from evals.datasets import available, get_adapter
from evals.repo_cache import RepoCache
from evals.report import aggregate, build_results, print_summary, write_results
from evals.scheduler import MAX_WORKERS, run_rollouts
from evals.types import RolloutResult, RolloutStatus, Task

console = Console()

DEFAULT_REPO_ROOT = Path("/tmp/nano-swebench/repos")


def _select(tasks: list[Task], instance_ids: str, repos: str, sample: int, seed: int) -> list[Task]:
    """Apply id/repo filters then random sampling (in that order)."""
    if instance_ids:
        wanted = {s.strip() for s in instance_ids.split(",") if s.strip()}
        tasks = [t for t in tasks if t.instance_id in wanted]
    if repos:
        wanted_repos = {s.strip() for s in repos.split(",") if s.strip()}
        tasks = [t for t in tasks if t.repo in wanted_repos]
    if sample and sample < len(tasks):
        rng = random.Random(seed)
        tasks = rng.sample(tasks, sample)
    return tasks


def _record_to_result(d: dict) -> RolloutResult:
    return RolloutResult(
        instance_id=d["instance_id"],
        repo=d["repo"],
        base_commit=d["base_commit"],
        status=RolloutStatus(d["status"]),
        model_patch=d.get("model_patch", ""),
        duration_s=d.get("duration_s", 0.0),
        error=d.get("error"),
        log_path=d.get("log_path"),
    )


def _result_to_record(r: RolloutResult) -> dict:
    return {
        "instance_id": r.instance_id,
        "repo": r.repo,
        "base_commit": r.base_commit,
        "status": r.status.value,
        "duration_s": round(r.duration_s, 2),
        "error": r.error,
        "log_path": r.log_path,
        "model_patch": r.model_patch,
    }


def _load_existing(records_path: Path) -> dict[str, RolloutResult]:
    if not records_path.exists():
        return {}
    done: dict[str, RolloutResult] = {}
    for line in records_path.read_text().splitlines():
        if line.strip():
            r = _record_to_result(json.loads(line))
            done[r.instance_id] = r
    return done


@click.command()
@click.option(
    "--dataset",
    default="swe-bench-lite",
    show_default=True,
    help=f"Dataset to run. Available: {', '.join(available())}.",
)
@click.option("--sample", type=int, default=0, help="Random subset size (0 = whole dataset).")
@click.option("--seed", type=int, default=0, show_default=True, help="Sampling seed.")
@click.option("--instance-ids", default="", help="Comma-separated instance ids to restrict to.")
@click.option("--repos", default="", help="Comma-separated repos (owner/name) to restrict to.")
@click.option(
    "--model", default=None, help="LiteLLM model string (default: nano-claude's default)."
)
@click.option("--max-turns", default=200, show_default=True, help="Agent max turns per task.")
@click.option(
    "--task-timeout", default=1800, show_default=True, help="Agent wall-clock budget (s)."
)
@click.option(
    "--workers",
    default=MAX_WORKERS,
    show_default=True,
    help=f"Concurrent rollout workers (capped at {MAX_WORKERS}).",
)
@click.option(
    "--repo-root",
    type=click.Path(),
    default=str(DEFAULT_REPO_ROOT),
    show_default=True,
    help="Where shared repo clones live.",
)
@click.option(
    "--output", type=click.Path(), default=None, help="Run directory (default: runs/<timestamp>)."
)
@click.option("--resume", is_flag=True, help="Skip instances already in the output dir's records.")
@click.option(
    "--no-strip-test-changes",
    is_flag=True,
    help="Keep agent edits to test files in the captured patch.",
)
@click.option(
    "--nano-bin", default="", help="Path to the nano-claude binary (default: autodetect)."
)
@click.option(
    "--eval/--no-eval",
    "do_eval",
    default=True,
    show_default=True,
    help="Run the Docker grading phase after rollout.",
)
@click.option("--eval-workers", default=4, show_default=True, help="Docker harness max workers.")
@click.option(
    "--eval-timeout", default=1800, show_default=True, help="Per-instance test timeout (s)."
)
@click.option("--run-id", default="", help="Harness run id (default: output dir name).")
def main(
    dataset: str,
    sample: int,
    seed: int,
    instance_ids: str,
    repos: str,
    model: str | None,
    max_turns: int,
    task_timeout: int,
    workers: int,
    repo_root: str,
    output: str | None,
    resume: bool,
    no_strip_test_changes: bool,
    nano_bin: str,
    do_eval: bool,
    eval_workers: int,
    eval_timeout: int,
    run_id: str,
) -> None:
    """Evaluate nano-claude-code on an agentic coding benchmark."""
    try:
        adapter = get_adapter(dataset)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from None

    run_dir = Path(output) if output else Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or run_dir.name
    records_path = run_dir / "rollout_records.jsonl"

    cfg = RolloutConfig(
        max_turns=max_turns,
        task_timeout=task_timeout,
        nano_bin=nano_bin,
        strip_test_changes=not no_strip_test_changes,
    )
    if model:
        cfg.model = model

    console.print(f"[bold]Loading[/bold] {dataset}...")
    tasks = _select(adapter.load(), instance_ids, repos, sample, seed)
    if not tasks:
        raise click.ClickException("No tasks selected.")

    done = _load_existing(records_path) if resume else {}
    pending = [t for t in tasks if t.instance_id not in done]
    n_repos = len({t.repo for t in pending})
    effective_workers = max(1, min(workers, MAX_WORKERS, n_repos)) if pending else 0
    console.print(
        f"[bold]{len(tasks)} task(s)[/bold] across {len({t.repo for t in tasks})} repo(s); "
        f"{len(done)} already done, {len(pending)} to run "
        f"on {effective_workers} worker(s). Model: {cfg.model}."
    )

    cache = RepoCache(Path(repo_root))
    log_dir = run_dir / "logs"

    # Append each record as it completes so a crash is resumable.
    records_file = records_path.open("a")

    def persist(result: RolloutResult) -> None:
        records_file.write(json.dumps(_result_to_record(result)) + "\n")
        records_file.flush()
        marker = "✓" if result.status is RolloutStatus.OK else f"! {result.status.value}"
        console.print(f"  [{marker}] {result.instance_id} ({result.duration_s:.0f}s)")

    try:
        new_results = run_rollouts(pending, cfg, cache, log_dir, workers, on_result=persist)
    finally:
        records_file.close()

    rollouts = list(done.values()) + new_results

    # Write predictions for the grader (one line per instance).
    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
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
    console.print(f"[dim]Predictions → {predictions_path}[/dim]")

    evals: dict = {}
    if do_eval:
        console.print("\n[bold]Evaluating[/bold] via the SWE-bench Docker harness...")
        gradable = [r.instance_id for r in rollouts if r.model_patch.strip()]
        if gradable:
            evals = adapter.evaluate(
                predictions_path, gradable, run_dir, run_id, eval_workers, eval_timeout
            )
        else:
            console.print("[yellow]No non-empty patches to grade.[/yellow]")

    results = build_results(rollouts, evals)
    summary = aggregate(results)
    write_results(run_dir, results, summary)
    print_summary(console, summary, evaluated=bool(evals))
    console.print(f"\n[dim]Full results in {run_dir}/[/dim]")


if __name__ == "__main__":
    main()
