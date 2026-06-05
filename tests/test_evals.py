"""Tests for the offline eval pipeline (no network / Docker required)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evals.config import RolloutConfig
from evals.datasets import available, get_adapter
from evals.repo_cache import RepoCache, changed_paths, strip_paths
from evals.report import aggregate, build_results
from evals.rollout import run_task
from evals.scheduler import MAX_WORKERS, partition_by_repo
from evals.types import EvalStatus, InstanceEval, RolloutStatus, Task


def _task(instance_id: str, repo: str) -> Task:
    return Task(instance_id=instance_id, repo=repo, base_commit="x", prompt="p")


# --- scheduler / partitioning -------------------------------------------------


def test_partition_keeps_each_repo_on_one_worker():
    tasks = (
        [_task(f"a{i}", "o/a") for i in range(3)]
        + [_task(f"b{i}", "o/b") for i in range(2)]
        + [_task("c0", "o/c")]
    )
    buckets = partition_by_repo(tasks, num_workers=2)

    # Every repo's tasks land in exactly one bucket (no repo split across workers).
    seen: dict[str, int] = {}
    for i, bucket in enumerate(buckets):
        for t in bucket:
            assert seen.setdefault(t.repo, i) == i
    # No task lost or duplicated.
    assert sum(len(b) for b in buckets) == len(tasks)


def test_run_rollouts_handles_no_pending_tasks(tmp_path):
    # --resume with everything already done -> empty task list must not crash
    # on ThreadPoolExecutor(max_workers=0).
    from evals.scheduler import run_rollouts

    cache = RepoCache(tmp_path / "repos")
    assert run_rollouts([], RolloutConfig(), cache, tmp_path / "logs", 5) == []


def test_partition_never_more_buckets_than_repos():
    tasks = [_task(f"a{i}", "o/a") for i in range(3)] + [_task("b0", "o/b")]
    buckets = partition_by_repo(tasks, num_workers=10)
    assert len(buckets) == 2  # only two repos


def test_partition_caps_at_max_workers():
    tasks = [_task(f"t{i}", f"o/r{i}") for i in range(20)]  # 20 distinct repos
    buckets = partition_by_repo(tasks, num_workers=99)
    assert len(buckets) == MAX_WORKERS


def test_partition_balances_load():
    # 6 small repos over 3 workers -> 2 each.
    tasks = [_task(f"t{i}", f"o/r{i}") for i in range(6)]
    buckets = partition_by_repo(tasks, num_workers=3)
    assert len(buckets) == 3
    assert all(len(b) == 2 for b in buckets)


# --- diff filtering -----------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/src/app.py b/src/app.py
index 111..222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1 @@
-old
+new
diff --git a/tests/test_app.py b/tests/test_app.py
index 333..444 100644
--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1 +1 @@
-old_test
+new_test
"""


def test_changed_paths_extracts_both_files():
    assert changed_paths(SAMPLE_DIFF) == {"src/app.py", "tests/test_app.py"}


def test_strip_paths_removes_only_named_section():
    stripped = strip_paths(SAMPLE_DIFF, {"tests/test_app.py"})
    assert "src/app.py" in stripped
    assert "tests/test_app.py" not in stripped
    assert "new_test" not in stripped
    assert "+new" in stripped


def test_strip_paths_noop_when_empty():
    assert strip_paths(SAMPLE_DIFF, set()) == SAMPLE_DIFF


# --- registry -----------------------------------------------------------------


def test_registry_has_swe_bench_lite():
    assert "swe-bench-lite" in available()


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_adapter("does-not-exist")


def test_parse_reports_reads_per_instance_verdicts(tmp_path):
    """The adapter reads each instance's report.json, not the top-level summary."""
    from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR

    from evals.datasets.swe_bench_lite import SweBenchLiteAdapter

    run_id, model = "r1", "nano__m"
    base = tmp_path / RUN_EVALUATION_LOG_DIR / run_id / "nano__m"
    # resolved instance
    (base / "good").mkdir(parents=True)
    (base / "good" / LOG_REPORT).write_text(
        json.dumps({"good": {"resolved": True, "tests_status": {"FAIL_TO_PASS": {}}}})
    )
    # graded-but-unresolved instance
    (base / "bad").mkdir(parents=True)
    (base / "bad" / LOG_REPORT).write_text(json.dumps({"bad": {"resolved": False}}))
    # "missing" instance has no report.json at all

    adapter = SweBenchLiteAdapter()
    out = adapter._parse_reports(tmp_path, run_id, model, ["good", "bad", "missing"])

    assert out["good"].status is EvalStatus.RESOLVED and out["good"].resolved
    assert out["bad"].status is EvalStatus.UNRESOLVED and not out["bad"].resolved
    assert out["missing"].status is EvalStatus.ERROR


# --- aggregation --------------------------------------------------------------


def test_aggregate_computes_resolve_rate_and_per_repo():
    from evals.types import RolloutResult

    rollouts = [
        RolloutResult("i1", "o/a", "c", RolloutStatus.OK, model_patch="diff"),
        RolloutResult("i2", "o/a", "c", RolloutStatus.OK, model_patch="diff"),
        RolloutResult("i3", "o/b", "c", RolloutStatus.EMPTY_PATCH),
    ]
    evals = {
        "i1": InstanceEval(EvalStatus.RESOLVED, resolved=True),
        "i2": InstanceEval(EvalStatus.UNRESOLVED),
        "i3": InstanceEval(EvalStatus.EMPTY_PATCH),
    }
    summary = aggregate(build_results(rollouts, evals))
    assert summary["total_instances"] == 3
    assert summary["resolved_instances"] == 1
    assert summary["resolve_rate"] == round(1 / 3, 4)
    assert summary["per_repo"]["o/a"]["resolved"] == 1
    assert summary["per_repo"]["o/a"]["total"] == 2


# --- repo_cache + rollout (offline, real git, fake agent) ---------------------


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=path,
        check=True,
        capture_output=True,
    )


def _make_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True)
    _git(repo_dir, "init", "-q")
    (repo_dir / "app.py").write_text("value = 1\n")
    (repo_dir / "tests").mkdir()
    (repo_dir / "tests" / "test_app.py").write_text("def test(): assert True\n")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-q", "-m", "base")
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _fake_agent(path: Path) -> str:
    """A stand-in for nano-claude: edits source + a test file, ignores stdin."""
    script = path / "fake-agent.sh"
    script.write_text(
        "#!/bin/sh\n"
        "cat > /dev/null\n"  # drain the prompt on stdin
        "echo 'value = 2' > app.py\n"
        "echo 'def test(): assert False' > tests/test_app.py\n"
    )
    script.chmod(0o755)
    return str(script)


def test_run_task_captures_patch_and_resets(tmp_path):
    root = tmp_path / "repos"
    repo_dir = root / "me__proj"  # matches RepoCache slug for "me/proj"
    base = _make_repo(repo_dir)
    cache = RepoCache(root)

    cfg = RolloutConfig(nano_bin=_fake_agent(tmp_path), task_timeout=30)
    task = Task(
        instance_id="me__proj-1",
        repo="me/proj",
        base_commit=base,
        prompt="fix it",
        extra={"test_patch": "diff --git a/tests/test_app.py b/tests/test_app.py\n"},
    )

    result = run_task(task, cache, cfg, tmp_path / "logs")

    assert result.status is RolloutStatus.OK
    # Source change captured...
    assert "value = 2" in result.model_patch
    assert "app.py" in result.model_patch
    # ...but the test-file edit was stripped (strip_test_changes default on).
    assert "test_app.py" not in result.model_patch
    # Working tree restored to pristine for the next task.
    assert (repo_dir / "app.py").read_text() == "value = 1\n"
    assert (repo_dir / "tests" / "test_app.py").read_text() == "def test(): assert True\n"


def test_run_task_reports_empty_patch_when_no_changes(tmp_path):
    root = tmp_path / "repos"
    repo_dir = root / "me__proj"
    base = _make_repo(repo_dir)
    cache = RepoCache(root)

    noop = tmp_path / "noop.sh"
    noop.write_text("#!/bin/sh\ncat > /dev/null\n")
    noop.chmod(0o755)

    cfg = RolloutConfig(nano_bin=str(noop), task_timeout=30)
    task = Task("me__proj-2", "me/proj", base, "do nothing")
    result = run_task(task, cache, cfg, tmp_path / "logs")
    assert result.status is RolloutStatus.EMPTY_PATCH
    assert result.model_patch.strip() == ""
