"""Memory directory resolution, the enablement gate, and path validation.

Memory is scoped to the repo's **canonical git root** (not cwd), so every
subdirectory and worktree of one repo shares a single memory directory while
different repos stay isolated. Resolution order:

1. ``NANO_CLAUDE_MEMORY_DIR`` env var, or ``memoryDirectory`` in settings — both
   run through :func:`validate_memory_path`.
2. ``<root>/projects/<sanitized-git-root>/memory/`` where the git root falls
   back to the project root (cwd) when there is no git repo.

This loosely mirrors ``src/memdir/paths.ts`` in the Claude Code source; the
validation here is a pathlib-native simplification of the upstream string checks.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from nano_claude.session.storage import DEFAULT_ROOT, sanitize_cwd

ENTRYPOINT = "MEMORY.md"
MAX_ENTRYPOINT_LINES = 200
# ~125 chars/line × 200 lines. Catches long-line indexes that slip the line cap.
MAX_ENTRYPOINT_BYTES = 25_000

# Truthy values for the disable env var.
_TRUTHY = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUTHY


def is_memory_enabled(settings: object | None = None) -> bool:
    """Whether the memory subsystem is active this session.

    Disabled when ``NANO_CLAUDE_DISABLE_MEMORY`` is truthy or settings set
    ``memoryEnabled: false``. Enabled by default otherwise.
    """
    if _is_truthy(os.environ.get("NANO_CLAUDE_DISABLE_MEMORY")):
        return False
    extra = getattr(settings, "extra", None)
    if isinstance(extra, dict) and extra.get("memoryEnabled") is False:
        return False
    return True


def validate_memory_path(raw: str | None) -> Path | None:
    """Normalize and validate a candidate memory-dir override.

    The memory dir is a read/write trust boundary, so we accept only a
    *concrete directory below a root* and reject everything else: NUL bytes,
    relative paths, and bare roots — the filesystem root ``/``, a Windows drive
    root ``C:\\``, or a UNC root ``\\\\server\\share``. ``~/...`` is expanded,
    but a bare ``~`` / ``~/`` / ``~/..`` (which resolves to ``$HOME`` or an
    ancestor) is rejected.

    Parsing is delegated to :class:`pathlib.Path`, which applies the rules of
    whichever OS is running: ``is_absolute()`` rejects relative paths, and
    ``len(parts) >= 2`` requires at least one component beneath the anchor,
    which is what distinguishes a real directory from a bare root on every
    platform. Returns the resolved absolute path, or ``None`` if unset/invalid.
    """
    if not raw or not raw.strip():
        return None
    candidate = raw.strip()
    if "\x00" in candidate:
        return None
    if candidate == "~" or candidate.startswith("~/"):
        rest = candidate[2:]  # everything after "~/"
        if not rest or os.path.normpath(rest) in (".", ".."):
            return None
        candidate = str(Path.home() / rest)
    path = Path(os.path.normpath(candidate))
    if not path.is_absolute() or len(path.parts) < 2:
        return None
    return path


def _memory_dir_override(settings: object | None) -> str | None:
    """The configured override, if any: env var first, then settings."""
    env = os.environ.get("NANO_CLAUDE_MEMORY_DIR")
    if env:
        return env
    extra = getattr(settings, "extra", None)
    if isinstance(extra, dict):
        value = extra.get("memoryDirectory")
        if isinstance(value, str):
            return value
    return None


def canonical_git_root(cwd: str) -> str | None:
    """The repo root shared by all worktrees, or ``None`` outside a repo.

    Uses ``git rev-parse --git-common-dir``: from any linked worktree this
    points at the *main* worktree's ``.git`` directory, whose parent is the
    canonical root. So all worktrees of one repo resolve to the same path.
    """
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    common = result.stdout.strip()
    if not common:
        return None
    common_path = Path(common)
    # ``--git-common-dir`` is the ``.git`` directory; its parent is the worktree
    # root. (For a bare repo it has no ``.git`` suffix — fall back to its parent.)
    root = common_path.parent if common_path.name == ".git" else common_path
    return str(root)


def memory_dir(
    cwd: str | None = None,
    settings: object | None = None,
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Resolve the memory directory for ``cwd`` (see module docstring)."""
    cwd = cwd or os.getcwd()
    override = validate_memory_path(_memory_dir_override(settings))
    if override is not None:
        return override
    base = canonical_git_root(cwd) or os.path.abspath(cwd)
    return root / "projects" / sanitize_cwd(base) / "memory"
