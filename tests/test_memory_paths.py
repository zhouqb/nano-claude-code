"""Tests for memory path resolution, the enablement gate, and validation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_claude.memory.paths import (
    canonical_git_root,
    is_memory_enabled,
    memory_dir,
    validate_memory_path,
)


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")


# --- validate_memory_path ---------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "   ", "relative/path", "..", "/", "/a", "~", "~/", "~/..", "a\x00b"]
)
def test_validate_rejects_dangerous_paths(bad):
    assert validate_memory_path(bad) is None


def test_validate_accepts_absolute(tmp_path):
    assert validate_memory_path(str(tmp_path)) == Path(str(tmp_path))


def test_validate_expands_tilde():
    result = validate_memory_path("~/some-mem-dir")
    assert result == Path.home() / "some-mem-dir"


# --- is_memory_enabled ------------------------------------------------------


def test_enabled_by_default(monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_DISABLE_MEMORY", raising=False)
    assert is_memory_enabled() is True


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("NANO_CLAUDE_DISABLE_MEMORY", "1")
    assert is_memory_enabled() is False


def test_disabled_by_settings(monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_DISABLE_MEMORY", raising=False)
    settings = SimpleNamespace(extra={"memoryEnabled": False})
    assert is_memory_enabled(settings) is False
    assert is_memory_enabled(SimpleNamespace(extra={})) is True


# --- memory_dir / canonical_git_root ----------------------------------------


def test_override_env_wins(monkeypatch, tmp_path):
    target = tmp_path / "custom-mem"
    monkeypatch.setenv("NANO_CLAUDE_MEMORY_DIR", str(target))
    assert memory_dir("/anywhere") == target


def test_subdirs_of_one_repo_share_a_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_MEMORY_DIR", raising=False)
    repo = tmp_path / "repo"
    _init_repo(repo)
    sub = repo / "pkg" / "deep"
    sub.mkdir(parents=True)
    root = tmp_path / "store"
    assert memory_dir(str(repo), root=root) == memory_dir(str(sub), root=root)


def test_worktrees_of_one_repo_share_a_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_MEMORY_DIR", raising=False)
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "f.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt))
    root = tmp_path / "store"
    assert memory_dir(str(repo), root=root) == memory_dir(str(wt), root=root)


def test_different_repos_are_isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_MEMORY_DIR", raising=False)
    a, b = tmp_path / "a", tmp_path / "b"
    _init_repo(a)
    _init_repo(b)
    root = tmp_path / "store"
    assert memory_dir(str(a), root=root) != memory_dir(str(b), root=root)


def test_no_git_falls_back_to_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("NANO_CLAUDE_MEMORY_DIR", raising=False)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert canonical_git_root(str(plain)) is None
    mdir = memory_dir(str(plain), root=tmp_path / "store")
    assert mdir.parent.name  # resolved to *some* per-cwd slug
    assert mdir.name == "memory"
