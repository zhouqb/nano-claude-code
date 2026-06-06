# evals — offline benchmark harness

Run nano-claude-code against agentic coding benchmarks. Currently supports
**SWE-bench Lite** and **SWE-bench Verified**; other datasets plug in via a small
adapter registry.

## Install

```bash
uv pip install -e ".[evals]"   # adds `datasets` + `swebench`
```

Everything runs in Docker (rollout *and* grading), so a working Docker daemon is
required.

## How it works

Both phases run inside the **official SWE-bench instance images**:

1. **Rollout** — for each task, start the instance container and run
   `nano-claude` *inside* it (one-shot `--stdin`, `bypassPermissions`, memory
   disabled) with the issue as the prompt. nano-claude installs into its own
   Python 3.12 venv at `/opt/nano` (via `uv`), while the agent's shell `PATH`
   points at the testbed conda env — so the agent runs the project's real
   interpreter and tests, however old. The `git diff` is captured as the
   `model_patch` (diffed against `HEAD`, with edits to graded test files
   stripped).
2. **Evaluation** — grade each patch with `swebench.harness.run_evaluation` on
   the same image (applies the patch + gold test patch, runs `FAIL_TO_PASS` /
   `PASS_TO_PASS`).

Images are **pre-built once** up front, then rollout + grade run on a **flat pool
of workers** (≤5). No repo affinity is needed — each task gets its own container,
so any worker can take any task. Grades run concurrently, each under its own
harness `run_id` (and `cache_level=instance`, so images persist and are reused).

## Usage

```bash
# Smoke run: 5 instances, 5 workers.
python -m evals.run --dataset swe-bench-verified --sample 5 --workers 5

# Whole dataset.
python -m evals.run --dataset swe-bench-verified --workers 5

# A single repo / specific instances.
python -m evals.run --repos django/django --sample 10
python -m evals.run --instance-ids astropy__astropy-12907,psf__requests-2317

# Batched coverage: non-overlapping 100s into one shared dir.
python -m evals.run --sample 100 --offset 0   --output runs/full --resume
python -m evals.run --sample 100 --offset 100 --output runs/full --resume

# Resume a crashed run (skips instances already in analysis.csv).
python -m evals.run --output runs/full --resume
```

Set the provider API key for `--model` first (e.g. `DEEPSEEK_API_KEY` for the
default `deepseek/deepseek-v4-flash`, or `--model anthropic/claude-sonnet-4-6`
with `ANTHROPIC_API_KEY`). Keys are forwarded into the container by name only.

### Key flags

| Flag | Meaning |
|---|---|
| `--dataset` | Benchmark (`swe-bench-lite` / `swe-bench-verified`). |
| `--sample N` / `--offset` / `--seed` | Window of size N over the seeded order, starting at `offset` (sample 0 = whole dataset). |
| `--instance-ids` / `--repos` | Restrict to specific instances or repos. |
| `--model` / `--max-turns` | Agent model and per-task turn cap. |
| `--task-timeout` / `--eval-timeout` | Agent budget / per-test timeout (s). |
| `--workers` | Worker pool size (capped at 5). |
| `--grade-workers` | Concurrent grades (0 = match `--workers`). |
| `--build-workers` | Concurrent image builds in the one-time prebuild (default 8). |
| `--eval / --no-eval` | Grade in Docker (default on; off = rollout only). |
| `--failure-study / --no-failure-study` | Bundle each miss's artifacts under `failures/<id>/` (default on). |
| `--resume` | Skip instances already in `analysis.csv`. |

## Outputs (under the run directory)

- `analysis.csv` — one row per instance: verdict, failure category, F2P/P2P
  counts, durations, patch + log paths. The results table.
- `predictions.jsonl` — `{instance_id, model_name_or_path, model_patch}`,
  consumable by the official harness.
- `results.jsonl` / `summary.json` — joined verdicts + resolve rate / per-repo
  breakdowns.
- `patches/<id>.diff`, `logs/` — captured patch and per-task agent + harness logs.
- `failures/<id>/` — (with `--failure-study`) each miss's agent log, patch,
  report, and test output, for offline review.

## Analyzing failures (`analyze`)

```bash
python -m evals.analyze runs/full            # -> analysis_summary.json, failures_review.md
python -m evals.analyze runs/full --merge-suggestions suggestions.json
```

`analyze` aggregates `analysis.csv`, bundles each failure's diff + failing tests
+ error output + transcript into `failures_review.md`, and can fold
externally-authored `{instance_id: {root_cause, improvement_suggestion}}` back
into the CSV. Suggestion-writing is left to a human or strong model, not an
automated small-model call.

## Adding a dataset

Implement `DatasetAdapter` (`load() -> list[Task]` and `evaluate(...)`) in
`evals/datasets/<name>.py`, then `register()` it in `evals/datasets/__init__.py`.
Rollout is fully generic — only loading and grading are dataset-specific.
