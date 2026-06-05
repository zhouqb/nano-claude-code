# evals — offline benchmark harness

Run nano-claude-code against agentic coding benchmarks. Currently supports
**SWE-bench Lite**; other datasets plug in via a small adapter registry.

## Install

```bash
uv pip install -e ".[evals]"   # adds `datasets` + `swebench`
```

The evaluation phase grades patches with the **official SWE-bench Docker
harness**, so a working Docker daemon is required for `--eval` (the default).
Use `--no-eval` to skip grading and only produce patches.

## How it works

Two phases:

1. **Rollout** (concurrent) — for each task: clone the target repo (once per
   repo, shared under `--repo-root`, default `/tmp/nano-swebench/repos`), check
   out its `base_commit`, run `nano-claude` one-shot (`--stdin`,
   `bypassPermissions`, memory disabled) with the issue as the prompt, capture
   `git diff` as the `model_patch`, then `git reset --hard && git clean -fdx`.
2. **Evaluation** (batched) — hand `predictions.jsonl` to
   `swebench.harness.run_evaluation`, which applies each patch + the gold test
   patch in Docker and runs `FAIL_TO_PASS` / `PASS_TO_PASS`.

### Concurrency & repo affinity

Tasks are grouped by repo and each repo is assigned to a single worker (≤5),
balanced by a longest-processing-time greedy pass. Because a repo's tasks share
one clone, they run **sequentially** within their worker and never concurrently
across workers. (On full Lite, django=114 tasks bound the best achievable
balance — one worker owns all of them.)

## Usage

```bash
# Quick smoke run: 5 instances, generate patches only (no Docker needed).
python -m evals.run --sample 5 --no-eval

# Full SWE-bench Lite, 5 rollout workers, then grade in Docker.
python -m evals.run --workers 5 --eval-workers 8

# A single repo / specific instances.
python -m evals.run --repos django/django --sample 10
python -m evals.run --instance-ids astropy__astropy-12907,psf__requests-2317

# Resume a crashed run (skips instances already recorded).
python -m evals.run --output runs/20260604-120000 --resume
```

Set the provider API key for `--model` first (e.g. `DEEPSEEK_API_KEY` for the
default `deepseek/deepseek-v4-flash`, or `--model anthropic/claude-sonnet-4-6`
with `ANTHROPIC_API_KEY`).

### Key flags

| Flag | Meaning |
|---|---|
| `--dataset` | Benchmark (default `swe-bench-lite`). |
| `--sample N` / `--seed` | Random subset of size N (0 = whole dataset). |
| `--instance-ids` / `--repos` | Restrict to specific instances or repos. |
| `--model` / `--max-turns` | Agent model and per-task turn cap. |
| `--task-timeout` | Per-task agent wall-clock budget (s). |
| `--workers` | Rollout workers (capped at 5). |
| `--eval / --no-eval` | Run the Docker grading phase (default on). |
| `--eval-workers` / `--eval-timeout` | Harness parallelism / per-test timeout. |
| `--resume` | Skip instances already in the run dir's records. |

## Outputs (under the run directory)

- `predictions.jsonl` — `{instance_id, model_name_or_path, model_patch}`,
  directly consumable by the official harness.
- `rollout_records.jsonl` — full per-task rollout record (status, patch,
  duration, log path); the resume source.
- `results.jsonl` — joined rollout + grading verdict per instance.
- `summary.json` — resolve rate + per-status / per-repo breakdowns.
- `logs/<instance_id>.log` — agent stdout/stderr per task.
- `swebench_report/` — the harness's own report JSON.

## Adding a dataset

Implement `DatasetAdapter` (`load() -> list[Task]` and `evaluate(...)`) in
`evals/datasets/<name>.py`, then `register()` it in `evals/datasets/__init__.py`.
The rollout phase is fully generic — only loading and grading are
dataset-specific.
