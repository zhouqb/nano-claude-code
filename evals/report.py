"""Join rollout + eval records, aggregate metrics, and render a summary."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.types import EvalStatus, InstanceEval, RolloutResult, TaskResult


def build_results(
    rollouts: list[RolloutResult],
    evals: dict[str, InstanceEval],
) -> list[TaskResult]:
    """Combine each rollout with its grading verdict (if evaluation ran)."""
    results: list[TaskResult] = []
    for r in rollouts:
        verdict = evals.get(r.instance_id)
        if verdict is None:
            eval_status, resolved = EvalStatus.SKIPPED, False
        else:
            eval_status, resolved = verdict.status, verdict.resolved
        results.append(
            TaskResult(
                instance_id=r.instance_id,
                repo=r.repo,
                rollout_status=r.status,
                eval_status=eval_status,
                resolved=resolved,
                duration_s=r.duration_s,
                patch_size=len(r.model_patch),
                error=r.error,
            )
        )
    return results


def aggregate(results: list[TaskResult]) -> dict:
    """Top-line metrics plus per-repo and per-status breakdowns."""
    total = len(results)
    resolved = sum(1 for r in results if r.resolved)
    by_rollout: dict[str, int] = defaultdict(int)
    by_eval: dict[str, int] = defaultdict(int)
    per_repo: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "resolved": 0})
    for r in results:
        by_rollout[r.rollout_status.value] += 1
        by_eval[r.eval_status.value] += 1
        per_repo[r.repo]["total"] += 1
        per_repo[r.repo]["resolved"] += int(r.resolved)

    return {
        "total_instances": total,
        "resolved_instances": resolved,
        "resolve_rate": round(resolved / total, 4) if total else 0.0,
        "rollout_status_counts": dict(by_rollout),
        "eval_status_counts": dict(by_eval),
        "per_repo": {
            repo: {
                **counts,
                "resolve_rate": round(counts["resolved"] / counts["total"], 4)
                if counts["total"]
                else 0.0,
            }
            for repo, counts in sorted(per_repo.items())
        },
    }


def write_results(run_dir: Path, results: list[TaskResult], summary: dict) -> None:
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r.to_json()) + "\n" for r in results))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def print_summary(console: Console, summary: dict, evaluated: bool) -> None:
    total = summary["total_instances"]
    if evaluated:
        pct = summary["resolve_rate"] * 100
        console.print(
            f"\n[bold]Resolved {summary['resolved_instances']}/{total} ({pct:.1f}%)[/bold]"
        )
    else:
        console.print(
            f"\n[bold]Rollout complete: {total} instance(s)[/bold] [dim](not evaluated)[/dim]"
        )

    rollout_table = Table(title="Rollout status", show_header=True)
    rollout_table.add_column("status")
    rollout_table.add_column("count", justify="right")
    for status, count in sorted(summary["rollout_status_counts"].items()):
        rollout_table.add_row(status, str(count))
    console.print(rollout_table)

    if evaluated:
        eval_table = Table(title="Eval status", show_header=True)
        eval_table.add_column("status")
        eval_table.add_column("count", justify="right")
        for status, count in sorted(summary["eval_status_counts"].items()):
            eval_table.add_row(status, str(count))
        console.print(eval_table)

        repo_table = Table(title="Per-repo resolve rate", show_header=True)
        repo_table.add_column("repo")
        repo_table.add_column("resolved", justify="right")
        repo_table.add_column("total", justify="right")
        repo_table.add_column("rate", justify="right")
        for repo, counts in summary["per_repo"].items():
            repo_table.add_row(
                repo,
                str(counts["resolved"]),
                str(counts["total"]),
                f"{counts['resolve_rate'] * 100:.0f}%",
            )
        console.print(repo_table)
