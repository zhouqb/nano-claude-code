"""Shared, per-repo git clones for the rollout phase.

Many tasks target the same GitHub repo (SWE-bench Lite: 114 django, 77 sympy,
...), so we clone each repo exactly once under a root directory and roll it to
whatever commit a task needs. Because the scheduler guarantees a repo is owned
by a single worker, the clone for a given repo is only ever touched by one
thread at a time — no per-repo locking is needed here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

GITHUB = "https://github.com/{repo}.git"


def _slug(repo: str) -> str:
    """``owner/name`` -> ``owner__name`` for a filesystem-safe directory."""
    return repo.replace("/", "__")


# Build artifacts an editable install (host-venv backend) drops in the clone.
# Seeded into .git/info/exclude so they never leak into the captured patch.
_LOCAL_EXCLUDE = ["*.egg-info/", ".eggs/", "build/", "__pycache__/", ".pytest_cache/"]


def _seed_local_exclude(repo_dir: Path) -> None:
    exclude = repo_dir / ".git" / "info" / "exclude"
    try:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text() if exclude.exists() else ""
        if "# nano-eval" not in existing:
            exclude.write_text(existing + "\n# nano-eval\n" + "\n".join(_LOCAL_EXCLUDE) + "\n")
    except OSError:
        pass


class GitError(RuntimeError):
    pass


def _git(repo_dir: Path, *args: str, timeout: int = 600) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


class RepoCache:
    """Manages one clone per repo under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def repo_dir(self, repo: str) -> Path:
        return self.root / _slug(repo)

    def ensure_clone(self, repo: str) -> Path:
        """Clone the repo if we haven't already; return its working directory."""
        path = self.repo_dir(repo)
        if (path / ".git").is_dir():
            return path
        # Clone into the slug dir. A full clone (not --depth) so arbitrary base
        # commits resolve without per-task fetches in the common case.
        subprocess.run(
            ["git", "clone", GITHUB.format(repo=repo), str(path)],
            capture_output=True,
            text=True,
            timeout=1800,
            check=True,
        )
        _seed_local_exclude(path)
        return path

    def checkout(self, repo: str, base_commit: str) -> Path:
        """Roll the shared clone to ``base_commit`` on a clean tree."""
        path = self.ensure_clone(repo)
        # Start clean so a previous task's leftovers never leak in.
        self.reset(path)
        try:
            _git(path, "checkout", "-f", base_commit)
        except GitError:
            # The commit may not be in the default fetch (rare); fetch it.
            _git(path, "fetch", "--quiet", "origin", base_commit, timeout=1800)
            _git(path, "checkout", "-f", base_commit)
        self.reset(path)
        return path

    @staticmethod
    def reset(repo_dir: Path) -> None:
        """Discard all changes and untracked files — a pristine working tree."""
        _git(repo_dir, "reset", "--hard")
        _git(repo_dir, "clean", "-fdx")

    @staticmethod
    def capture_patch(repo_dir: Path, base_commit: str) -> str:
        """Diff the working tree (incl. new files) against ``base_commit``.

        ``git add -A`` stages new/modified/deleted tracked files (honoring
        .gitignore, so build artifacts are excluded), then we diff the index
        against the base commit to produce a clean, apply-able patch.
        """
        _git(repo_dir, "add", "-A")
        return _git(repo_dir, "diff", "--cached", base_commit)


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
