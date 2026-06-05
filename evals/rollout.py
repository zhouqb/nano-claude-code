"""Run nano-claude one-shot on a single task and capture its patch."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from evals.config import RolloutConfig
from evals.repo_cache import RepoCache, changed_paths, strip_paths
from evals.types import RolloutResult, RolloutStatus, Task


def _run_agent(task: Task, repo_dir: Path, cfg: RolloutConfig, log_path: Path) -> None:
    """Invoke the CLI in one-shot mode, killing the whole group on timeout.

    nano-claude's one-shot mode defaults to ``bypassPermissions`` so the agent
    runs Bash/Edit/Write without prompting. We disable memory so runs stay
    isolated and reproducible, and route stdout/stderr to a per-task log.
    """
    env = {
        **os.environ,
        "NANO_CLAUDE_DISABLE_MEMORY": "1",
    }
    cmd = [
        cfg.resolved_bin(),
        "--stdin",
        "--model",
        cfg.model,
        "--max-turns",
        str(cfg.max_turns),
        "--permission-mode",
        "bypassPermissions",
    ]
    with log_path.open("w") as log:
        # start_new_session so a timeout can kill the agent *and* any child
        # processes it spawned (Bash tool subprocesses) via the process group.
        proc = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            env=env,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            proc.communicate(input=task.prompt, timeout=cfg.task_timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"nano-claude exited with code {proc.returncode}")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_task(task: Task, cache: RepoCache, cfg: RolloutConfig, log_dir: Path) -> RolloutResult:
    """Check out, run the agent, capture the diff, and reset the clone."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.instance_id}.log"
    started = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - started

    try:
        repo_dir = cache.checkout(task.repo, task.base_commit)
    except Exception as exc:  # noqa: BLE001 - clone/checkout failure ends this task
        return RolloutResult(
            task.instance_id,
            task.repo,
            task.base_commit,
            RolloutStatus.ERROR,
            error=f"checkout failed: {exc}",
            duration_s=elapsed(),
        )

    status = RolloutStatus.OK
    error: str | None = None
    try:
        _run_agent(task, repo_dir, cfg, log_path)
    except subprocess.TimeoutExpired:
        status = RolloutStatus.TIMEOUT
        error = f"agent exceeded {cfg.task_timeout}s"
    except Exception as exc:  # noqa: BLE001 - surface agent failure, still capture patch
        status = RolloutStatus.ERROR
        error = str(exc)

    # Capture whatever the agent changed even on timeout/error — a partial fix
    # is still gradable. Then always restore a pristine tree for the next task.
    patch = ""
    try:
        patch = cache.capture_patch(repo_dir, task.base_commit)
        if cfg.strip_test_changes:
            patch = strip_paths(patch, _test_paths(task))
    except Exception as exc:  # noqa: BLE001
        if status is RolloutStatus.OK:
            status = RolloutStatus.ERROR
            error = f"patch capture failed: {exc}"
    finally:
        try:
            cache.reset(repo_dir)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass

    if status is RolloutStatus.OK and not patch.strip():
        status = RolloutStatus.EMPTY_PATCH

    return RolloutResult(
        instance_id=task.instance_id,
        repo=task.repo,
        base_commit=task.base_commit,
        status=status,
        model_patch=patch,
        duration_s=elapsed(),
        error=error,
        log_path=str(log_path),
    )


def _test_paths(task: Task) -> set[str]:
    """Files touched by the dataset's gold test patch, if any."""
    test_patch = task.extra.get("test_patch", "")
    return changed_paths(test_patch) if test_patch else set()
