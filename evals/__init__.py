"""Offline evaluation pipeline for nano-claude-code.

A two-phase harness for running nano-claude against agentic coding benchmarks:

1. **Rollout** (``evals.scheduler`` / ``evals.rollout``) — for each task, check
   out the target repo at its base commit, run nano-claude one-shot, capture the
   resulting diff, and reset the clone. Concurrent across repos (<=5 workers)
   with repo-affinity so a single repo's tasks never run in parallel.
2. **Evaluation** (``evals.datasets`` adapters) — grade the captured patches.
   For SWE-bench this defers to the official Docker harness.

Currently only SWE-bench Lite is wired up; new datasets plug in via the
``DatasetAdapter`` registry in ``evals.datasets``.
"""
