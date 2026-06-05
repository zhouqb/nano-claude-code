"""Host-side per-(repo, version) virtualenv provider for the rollout.

Gives the agent a *real* environment to run tests in — without Docker — for the
SWE-bench instances that are feasible on this host: Python >= 3.8 (uv can supply
the interpreter) and no compiled native dependencies (numpy/scipy/cython/... do
not reliably build on macOS arm64 at the old pinned versions).

We reuse SWE-bench's own per-(repo, version) recipe from
``MAP_REPO_VERSION_TO_SPECS`` (Python version, extra packages, test command), and
install the project **editable** so the agent's edits in the clone are live for
test runs. Venvs are cached per (repo, version) and live outside the clone, so a
``git clean`` of the clone never disturbs them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from evals.types import Task

# Repos whose *own* package is a C/Cython extension, so an editable install
# compiles from source — old pinned versions don't build on macOS arm64. (A repo
# that merely *depends* on numpy/scipy/pandas is fine: pip fetches arm64 wheels,
# and a missing old wheel just makes the venv build fail -> graceful fallback.)
_COMPILED_EXTENSION_REPOS = {
    "scikit-learn/scikit-learn",
    "matplotlib/matplotlib",
    "astropy/astropy",
    "numpy/numpy",
    "pandas/pandas",
    "python-pillow/Pillow",
}


def _spec(task: Task) -> dict:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    return MAP_REPO_VERSION_TO_SPECS.get(task.repo, {}).get(task.extra.get("version", ""), {})


def _pkg_tokens(spec: dict) -> list[str]:
    """All declared extra packages (pinned ``pip_packages`` + ``packages``)."""
    out: list[str] = []
    for key in ("pip_packages", "packages"):
        val = spec.get(key)
        if isinstance(val, (list, tuple)):
            out += [str(v) for v in val]
        elif isinstance(val, str):
            out += val.split()
    return out


def _install_tokens(spec: dict) -> list[str]:
    """Installable package specs from a swebench env spec.

    Drops non-package sentinels: ``python`` and requirements-file references
    (e.g. django records its deps as ``packages: "requirements.txt"``, which is
    not an installable name — the editable install pulls runtime deps anyway).
    """
    return [
        t
        for t in _pkg_tokens(spec)
        if t.lower() != "python" and not t.lower().endswith((".txt", ".cfg", ".ini"))
    ]


def _py_ok(py: str) -> bool:
    try:
        return tuple(int(p) for p in py.split(".")) >= (3, 8)
    except ValueError:
        return False


def is_feasible(task: Task) -> bool:
    """True if this instance can plausibly get a working host venv (no Docker).

    Hard requirements: an installable interpreter (Python >= 3.8) and a project
    that isn't itself a native extension. Dependency wheels (numpy/pandas/...) are
    left to pip; if an old one is missing the build fails and we fall back.
    """
    spec = _spec(task)
    if not spec or not _py_ok(spec.get("python", "")):
        return False
    if task.repo in _COMPILED_EXTENSION_REPOS:
        return False
    return True


@dataclass
class VenvInfo:
    bin_dir: Path
    test_cmd: str


class VenvCache:
    """Builds and caches one editable venv per (repo, version)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, task: Task) -> Path:
        slug = task.repo.replace("/", "__")
        return self.root / f"{slug}__{task.extra.get('version', 'x')}"

    def ensure(self, task: Task, clone_dir: Path) -> VenvInfo | None:
        """Return a ready venv for this instance, building it once. None on failure."""
        spec = _spec(task)
        test_cmd = str(spec.get("test_cmd") or "python -m pytest")
        vdir = self._dir(task)
        if (vdir / ".ready").exists():
            return VenvInfo(vdir / "bin", test_cmd)

        py = spec.get("python", "3.11")
        pip = ["uv", "pip", "install", "--python", str(vdir / "bin" / "python"), "-q"]
        try:
            shutil.rmtree(vdir, ignore_errors=True)
            subprocess.run(
                ["uv", "venv", "--python", py, str(vdir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
            tokens = _install_tokens(spec)
            if tokens:
                subprocess.run(
                    pip + tokens, check=True, capture_output=True, text=True, timeout=1800
                )
            # A runner the agent can always rely on, plus the project (editable).
            subprocess.run(
                pip + ["pytest"], check=True, capture_output=True, text=True, timeout=900
            )
            subprocess.run(
                pip + ["-e", str(clone_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            shutil.rmtree(vdir, ignore_errors=True)
            return None
        (vdir / ".ready").write_text("ok")
        return VenvInfo(vdir / "bin", test_cmd)


def filter_feasible(tasks: list[Task]) -> list[Task]:
    """Keep only instances that can get a working host venv on this machine."""
    return [t for t in tasks if is_feasible(t)]


def make_env_provider(backend: str, venv_root: Path):
    """Build the rollout env provider for a backend ('host' -> None)."""
    if backend == "host-venv":
        return VenvEnvProvider(VenvCache(venv_root))
    return None


class VenvEnvProvider:
    """Rollout EnvProvider backed by :class:`VenvCache`."""

    def __init__(self, cache: VenvCache) -> None:
        self.cache = cache

    def prepare(self, task: Task, repo_dir: Path) -> tuple[dict[str, str], str] | None:
        from evals.prompts import verify_addendum

        info = self.cache.ensure(task, repo_dir)
        if info is None:
            return None
        env = {
            "PATH": f"{info.bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "VIRTUAL_ENV": str(info.bin_dir.parent),
        }
        return env, verify_addendum(info.test_cmd)
