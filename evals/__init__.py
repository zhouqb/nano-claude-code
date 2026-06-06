"""Offline evaluation pipeline for nano-claude-code.

A Docker-based harness for running nano-claude against agentic coding benchmarks.
For each task, nano-claude rolls out *inside* the benchmark's instance container
(exact interpreter + prebuilt deps, can run the project's tests), and the
captured patch is graded with the same image. Images are pre-built up front;
rollout + grade run on a flat pool of workers (<=5).

The entry point is ``evals.run``. Datasets plug in via the ``DatasetAdapter``
registry in ``evals.datasets`` (currently SWE-bench Lite + Verified).
"""
