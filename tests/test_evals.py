"""Tests for the offline eval pipeline (no network / Docker required)."""

from __future__ import annotations

import csv
import json

import pytest

from evals.datasets import available, get_adapter
from evals.patch_utils import changed_paths, is_test_path, strip_paths
from evals.report import aggregate, build_results
from evals.types import EvalStatus, InstanceEval, RolloutStatus

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


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_app.py",
        "sympy/core/tests/test_basic.py",
        "lib/matplotlib/tests/test_text.py",
        "tests/conftest.py",
        "foo/bar_test.py",
        "tests/helpers.py",  # non-test-named, but under a tests/ tree
    ],
)
def test_is_test_path_true(path):
    assert is_test_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "src/app.py",
        "sympy/core/basic.py",
        # Importable SOURCE trees that must NOT be mistaken for tests:
        "django/test/utils.py",  # django's test framework is shipped source
        "django/test/client.py",
        "lib/matplotlib/testing/decorators.py",  # matplotlib.testing is source
        "pkg/contest.py",  # not conftest
        "tests/data/fixture.json",  # not a .py file
    ],
)
def test_is_test_path_false(path):
    assert is_test_path(path) is False


def test_strip_reverts_agent_test_files_but_keeps_source():
    # The agent's source fix survives; its new test file is reverted out.
    diff = (
        "diff --git a/django/test/utils.py b/django/test/utils.py\n"
        "index 1..2 100644\n--- a/django/test/utils.py\n+++ b/django/test/utils.py\n"
        "@@ -1 +1 @@\n-old\n+fixed\n"
        "diff --git a/tests/test_new.py b/tests/test_new.py\n"
        "new file mode 100644\n--- /dev/null\n+++ b/tests/test_new.py\n"
        "@@ -0,0 +1 @@\n+def test_repro(): assert True\n"
    )
    drop = {p for p in changed_paths(diff) if is_test_path(p)}
    stripped = strip_paths(diff, drop)
    assert "django/test/utils.py" in stripped  # source fix kept
    assert "+fixed" in stripped
    assert "tests/test_new.py" not in stripped  # agent test reverted
    assert "test_repro" not in stripped


# --- analyze ------------------------------------------------------------------


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_analyze_aggregate_and_merge(tmp_path):
    from evals.analyze import aggregate as analyze_aggregate
    from evals.analyze import merge_suggestions

    rows = [
        {
            "instance_id": "a-1",
            "repo": "o/a",
            "resolved": "True",
            "failure_category": "",
            "env_ready": "True",
            "improvement_suggestion": "",
        },
        {
            "instance_id": "a-2",
            "repo": "o/a",
            "resolved": "False",
            "failure_category": "tests_failed",
            "env_ready": "True",
            "improvement_suggestion": "",
        },
        {
            "instance_id": "b-1",
            "repo": "o/b",
            "resolved": "False",
            "failure_category": "empty_patch",
            "env_ready": "False",
            "improvement_suggestion": "",
        },
    ]
    csv_path = tmp_path / "analysis.csv"
    _write_csv(csv_path, rows)

    summary = analyze_aggregate(rows)
    assert summary["resolved_instances"] == 1
    assert summary["resolve_rate"] == round(1 / 3, 4)
    assert summary["failure_categories"] == {"tests_failed": 1, "empty_patch": 1}
    assert summary["env_ready_counts"] == {"True": 2, "False": 1}
    assert summary["per_repo"]["o/a"]["resolved"] == 1

    (tmp_path / "sugg.json").write_text(
        json.dumps(
            {
                "a-2": {"root_cause": "rc", "improvement_suggestion": "do X"},
            }
        )
    )
    merged, over = merge_suggestions(tmp_path, tmp_path / "sugg.json")
    assert merged == 1 and over == []
    by_id = {r["instance_id"]: r for r in csv.DictReader(csv_path.open())}
    assert by_id["a-2"]["root_cause"] == "rc"
    assert by_id["a-2"]["improvement_suggestion"] == "do X"
    assert by_id["a-1"]["improvement_suggestion"] == ""  # untouched


def test_analyze_build_review(tmp_path):
    from evals.analyze import build_review

    (tmp_path / "failures" / "a-2").mkdir(parents=True)
    (tmp_path / "failures" / "a-2" / "model_patch.diff").write_text(
        "diff --git a/src/x.py b/src/x.py\n+CHANGED\n"
    )
    (tmp_path / "failures" / "a-2" / "test_output.txt").write_text("E   AssertionError: nope\n")
    rows = [
        {"instance_id": "a-1", "repo": "o/a", "resolved": "True", "failure_category": ""},
        {
            "instance_id": "a-2",
            "repo": "o/a",
            "resolved": "False",
            "failure_category": "tests_failed",
        },
    ]
    md = build_review(tmp_path, rows).read_text()
    assert "a-2" in md and "CHANGED" in md and "AssertionError" in md
    assert "a-1" not in md  # passes aren't in the review


# --- registry -----------------------------------------------------------------


def test_registry_has_swe_bench_lite():
    assert "swe-bench-lite" in available()


def test_verified_adapter_registered():
    assert "swe-bench-verified" in available()


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
