"""Repo-affinity worker scheduling for the rollout phase.

Constraints we honor:
- Tasks on the same repo share one git clone, so they must run **sequentially**.
- Therefore every task of a repo is assigned to the **same worker**; workers run
  in parallel, but never two tasks of one repo at once.
- At most ``MAX_WORKERS`` workers.

Balancing is longest-processing-time greedy bin-packing over repo groups: assign
the largest remaining repo to the currently least-loaded worker. The biggest
single repo bounds the best achievable balance (django=114 on full Lite).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.config import RolloutConfig
from evals.repo_cache import RepoCache
from evals.rollout import run_task
from evals.types import RolloutResult, Task

MAX_WORKERS = 5


def partition_by_repo(tasks: list[Task], num_workers: int) -> list[list[Task]]:
    """Split tasks into per-worker lists, keeping each repo on one worker.

    Returns a list of task-lists (one per worker, repo-contiguous). The number
    of buckets is ``min(num_workers, MAX_WORKERS, #repos)`` — never more workers
    than repos, since a repo cannot be split.
    """
    groups: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        groups[task.repo].append(task)

    n = max(1, min(num_workers, MAX_WORKERS, len(groups)))
    buckets: list[list[Task]] = [[] for _ in range(n)]
    loads = [0] * n

    # Largest repo groups first -> assign to the least-loaded bucket (LPT).
    for _repo, group in sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True):
        i = loads.index(min(loads))
        buckets[i].extend(group)
        loads[i] += len(group)

    return [b for b in buckets if b]


def run_rollouts(
    tasks: list[Task],
    cfg: RolloutConfig,
    cache: RepoCache,
    log_dir: Path,
    num_workers: int,
    on_result: Callable[[RolloutResult], None] | None = None,
    env_provider=None,
) -> list[RolloutResult]:
    """Run every task, concurrently across repo-partitioned workers.

    ``on_result`` (if given) is called as each task finishes, under a lock, so
    callers can persist predictions/records incrementally. The shared ``cache``
    is safe across threads because repos are disjoint across workers.
    """
    buckets = partition_by_repo(tasks, num_workers)
    if not buckets:  # nothing to do (e.g. --resume with everything already done)
        return []
    results: list[RolloutResult] = []
    lock = threading.Lock()

    def worker(bucket: list[Task]) -> None:
        for task in bucket:
            result = run_task(task, cache, cfg, log_dir, env_provider)
            with lock:
                results.append(result)
                if on_result is not None:
                    on_result(result)

    with ThreadPoolExecutor(max_workers=len(buckets)) as pool:
        futures = [pool.submit(worker, bucket) for bucket in buckets]
        for future in futures:
            future.result()  # surface any worker-thread exception

    return results
