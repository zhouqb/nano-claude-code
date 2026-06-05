"""Analyze a completed study run (the CSV + artifacts written by ``failure_study``).

Three jobs, all the *mechanical* parts of analysis — the judgment (writing the
per-failure improvement suggestions) stays with a human or a strong model:

1. **Aggregate** — resolve rate, per-repo, failure-category and env-readiness
   breakdowns. Printed and written to ``analysis_summary.json``.
2. **Review bundle** — ``failures_review.md`` with, per failure, the failing
   tests + the agent's diff + the harness error output + the agent transcript
   tail. This is the doc a reviewer reads to write suggestions.
3. **Merge suggestions** — fold externally-authored ``{instance_id: {root_cause,
   improvement_suggestion}}`` back into the CSV (replaces hardcoded scratch).

Run:
    python -m evals.analyze runs/study-50-0
    python -m evals.analyze runs/study-50-0 --merge-suggestions suggestions.json
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_ERROR_LINE = re.compile(
    r"Error|assert|Assertion|Traceback|FAILED|!=|Expected|raise|Mismatch|not equal|did not"
)


def load_rows(run_dir: Path) -> list[dict]:
    csv_path = Path(run_dir) / "analysis.csv"
    if not csv_path.is_file():
        raise click.ClickException(f"No analysis.csv in {run_dir}")
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _is_resolved(row: dict) -> bool:
    return str(row.get("resolved", "")).lower() == "true"


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    resolved = sum(1 for r in rows if _is_resolved(r))
    failures = [r for r in rows if not _is_resolved(r)]
    per_repo: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "resolved": 0})
    for r in rows:
        per_repo[r["repo"]]["total"] += 1
        per_repo[r["repo"]]["resolved"] += int(_is_resolved(r))
    return {
        "total_instances": total,
        "resolved_instances": resolved,
        "resolve_rate": round(resolved / total, 4) if total else 0.0,
        "failure_categories": dict(Counter(r.get("failure_category", "") for r in failures)),
        "env_ready_counts": dict(Counter(r.get("env_ready", "") for r in rows)),
        "per_repo": {
            repo: {
                **c,
                "resolve_rate": round(c["resolved"] / c["total"], 4) if c["total"] else 0.0,
            }
            for repo, c in sorted(per_repo.items())
        },
    }


def _read(path: Path, tail: int | None = None) -> str:
    try:
        text = _ANSI.sub("", path.read_text(errors="replace"))
    except OSError:
        return ""
    return text[-tail:] if tail else text


def _failing_tests(failure_dir: Path, iid: str) -> list[str]:
    report = failure_dir / "report.json"
    if not report.is_file():
        return []
    try:
        verdict = json.loads(report.read_text())[iid]
        return verdict.get("tests_status", {}).get("FAIL_TO_PASS", {}).get("failure", [])
    except (OSError, json.JSONDecodeError, KeyError):
        return []


def _error_excerpt(text: str, limit: int = 40) -> str:
    lines = [line for line in text.splitlines() if _ERROR_LINE.search(line)]
    return "\n".join(lines[-limit:]) if lines else "\n".join(text.splitlines()[-20:])


def build_review(run_dir: Path, rows: list[dict]) -> Path:
    """Write failures_review.md: one section per failure, for a reviewer."""
    run_dir = Path(run_dir)
    out: list[str] = ["# Failures review\n"]
    failures = [r for r in rows if not _is_resolved(r)]
    out.append(f"{len(failures)} failure(s) of {len(rows)} cases.\n")
    for r in failures:
        iid = r["instance_id"]
        fdir = run_dir / "failures" / iid
        patch = _read(fdir / "model_patch.diff") or _read(Path(r.get("patch_file", "")))
        test_out = _read(fdir / "test_output.txt") or _read(Path(r.get("test_output", "")))
        agent = _read(fdir / "agent.log", tail=3000) or _read(
            Path(r.get("agent_log", "")), tail=3000
        )
        out += [
            f"\n## {iid}",
            f"- repo: `{r['repo']}` | category: `{r.get('failure_category', '')}` | "
            f"F2P {r.get('f2p_passed', '?')}/{r.get('f2p_total', '?')} | "
            f"p2p_failed {r.get('p2p_failed', '?')} | env_ready={r.get('env_ready', '')}",
            f"- failing tests: {_failing_tests(fdir, iid)[:8]}",
            "\n### Patch\n```diff",
            patch.strip()[:2500] or "(empty)",
            "```",
            "\n### Test output (errors)\n```",
            _error_excerpt(test_out)[:2000] or "(none)",
            "```",
            "\n### Agent transcript (tail)\n```",
            agent.strip()[-2000:] or "(none)",
            "```",
        ]
    path = run_dir / "failures_review.md"
    path.write_text("\n".join(out))
    return path


def merge_suggestions(run_dir: Path, suggestions_path: Path) -> tuple[int, list[str]]:
    """Fold {instance_id: {root_cause, improvement_suggestion}} into the CSV.

    Returns (n_merged, over_100_word_instance_ids).
    """
    run_dir = Path(run_dir)
    data = json.loads(Path(suggestions_path).read_text())
    csv_path = run_dir / "analysis.csv"
    rows = load_rows(run_dir)
    fields = list(rows[0].keys())
    if "root_cause" not in fields:
        anchor = (
            fields.index("improvement_suggestion")
            if "improvement_suggestion" in fields
            else len(fields)
        )
        fields.insert(anchor, "root_cause")
    if "improvement_suggestion" not in fields:
        fields.append("improvement_suggestion")

    over: list[str] = []
    merged = 0
    for r in rows:
        r.setdefault("root_cause", "")
        r.setdefault("improvement_suggestion", "")
        entry = data.get(r["instance_id"])
        if not entry:
            continue
        r["root_cause"] = entry.get("root_cause", r["root_cause"])
        sugg = entry.get("improvement_suggestion", r["improvement_suggestion"])
        r["improvement_suggestion"] = sugg
        merged += 1
        if len(sugg.split()) > 100:
            over.append(r["instance_id"])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return merged, over


def print_summary(summary: dict) -> None:
    rate = summary["resolve_rate"] * 100
    console.print(
        f"\n[bold]Resolved {summary['resolved_instances']}/{summary['total_instances']} "
        f"({rate:.1f}%)[/bold]"
    )
    if summary["failure_categories"]:
        console.print(f"[dim]failure categories:[/dim] {summary['failure_categories']}")
    if any(k for k in summary["env_ready_counts"]):
        console.print(f"[dim]env_ready:[/dim] {summary['env_ready_counts']}")
    table = Table(title="Per-repo resolve rate")
    table.add_column("repo")
    table.add_column("resolved", justify="right")
    table.add_column("total", justify="right")
    table.add_column("rate", justify="right")
    for repo, c in summary["per_repo"].items():
        table.add_row(repo, str(c["resolved"]), str(c["total"]), f"{c['resolve_rate'] * 100:.0f}%")
    console.print(table)


@click.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--review/--no-review", default=True, help="Write failures_review.md.")
@click.option(
    "--merge-suggestions",
    "suggestions",
    default="",
    help="JSON {instance_id: {root_cause, improvement_suggestion}} to fold into the CSV.",
)
def main(run_dir: str, review: bool, suggestions: str) -> None:
    """Aggregate + bundle a study run for review; optionally merge suggestions."""
    run_path = Path(run_dir)
    rows = load_rows(run_path)
    summary = aggregate(rows)
    (run_path / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)

    if review:
        path = build_review(run_path, rows)
        console.print(f"[dim]Review bundle → {path}[/dim]")
    if suggestions:
        merged, over = merge_suggestions(run_path, Path(suggestions))
        console.print(f"[green]Merged {merged} suggestion(s) into the CSV.[/green]")
        if over:
            console.print(f"[yellow]Over 100 words: {over}[/yellow]")


if __name__ == "__main__":
    main()
