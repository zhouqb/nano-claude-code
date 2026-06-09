"""Helpers for working with the captured ``git diff`` (the model patch).

The docker rollout backend captures the agent's changes as a unified diff and
needs to (a) keep build artifacts / scratch databases out of it and (b) strip
out edits to graded test files. These pure functions do that; they have no git
or filesystem state of their own.
"""

from __future__ import annotations

# Files we never want in the captured patch, seeded into .git/info/exclude:
# scratch databases the agent's reproduction scripts tend to drop in the repo
# root (e.g. django leaves ``other_N.sqlite3`` files) and build artifacts, which
# would otherwise pollute the model_patch.
_LOCAL_EXCLUDE = [
    "*.egg-info/",
    ".eggs/",
    "build/",
    "__pycache__/",
    ".pytest_cache/",
    "*.sqlite3",
    "*.sqlite",
    "*.db",
    # Scratch reproduction scripts the agent is told to write at the repo root —
    # keep them out of the captured patch even if it forgets to use /tmp.
    "repro*.py",
    "reproduce*.py",
]

# Directory name that marks a Python test tree across the SWE-bench repos
# (django top-level ``tests/``; ``<pkg>/.../tests/`` in sympy, scikit-learn,
# astropy, xarray, matplotlib; ``tests/`` in sphinx, pylint). Deliberately the
# *plural* only: ``django/test/`` and ``matplotlib/testing/`` are importable
# SOURCE trees, so matching ``test``/``testing`` would revert real fixes.
_TEST_DIR_PARTS = frozenset({"tests"})


def is_test_path(path: str) -> bool:
    """Whether ``path`` is a Python test file whose edits should be reverted out
    of the graded patch (so the agent may write tests to verify itself without
    those changes ever reaching the grader).

    Conservative by design — it matches the standard pytest/unittest conventions
    the SWE-bench repos follow and never classifies package source as a test: a
    path qualifies only when it is a ``.py`` whose basename looks like a test
    (``test_*.py`` / ``*_test.py`` / ``conftest.py``) OR which sits under a
    ``tests/`` directory.
    """
    parts = path.split("/")
    base = parts[-1]
    if not base.endswith(".py"):
        return False
    stem = base[:-3]
    if base == "conftest.py" or stem.startswith("test_") or stem.endswith("_test"):
        return True
    return any(part in _TEST_DIR_PARTS for part in parts[:-1])


def changed_paths(patch: str) -> set[str]:
    """File paths touched by a unified diff (from its ``diff --git`` headers)."""
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            # "diff --git a/<path> b/<path>"
            parts = line.split(" ")
            if len(parts) >= 4 and parts[3].startswith("b/"):
                paths.add(parts[3][2:])
    return paths


def strip_paths(patch: str, drop: set[str]) -> str:
    """Remove whole-file sections of a unified diff for paths in ``drop``.

    Used to discard any edits the agent made to graded test files, which would
    otherwise collide with the gold test patch the harness applies.
    """
    if not drop:
        return patch
    out: list[str] = []
    keep = True
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            parts = line.split(" ")
            target = parts[3][2:] if len(parts) >= 4 and parts[3].startswith("b/") else ""
            keep = target.rstrip("\n") not in drop
        if keep:
            out.append(line)
    return "".join(out)
