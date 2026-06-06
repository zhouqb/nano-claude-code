"""Docker rollout backend: run nano-claude *inside* each SWE-bench container.

The host and host-venv backends can only cover the ~60% of Verified instances
whose environment we can reproduce on macOS arm64. The remaining ~40% need a
compiled-dependency build (scikit-learn/matplotlib/astropy/numpy/pandas/Pillow)
or an old Python the host can't supply. SWE-bench's own instance images already
bake those environments, so here we run the agent *inside* the instance
container — it gets the exact interpreter and pre-built deps and can run the
project's tests.

Design notes:
- nano-claude needs Python >= 3.12 and heavy deps, while the testbed conda env
  ranges 3.6-3.11. So we install nano-claude into its *own* interpreter at
  ``/opt/nano`` (via ``uv``) and never touch the testbed env. The agent's shell
  PATH is prefixed with the testbed conda bin so its ``python``/``pytest`` hit
  the real environment.
- The uv cache + downloaded interpreters are bind-mounted from the host, so the
  3.12 download and the wheel set are fetched once and reused across containers.
- API keys are forwarded by *name only* (``docker exec -e NAME``), so secrets
  never appear in argv or logs.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from evals.config import RolloutConfig
from evals.prompts import verify_addendum
from evals.repo_cache import _LOCAL_EXCLUDE, changed_paths, strip_paths
from evals.types import RolloutResult, RolloutStatus, Task

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TESTBED = "/testbed"
NANO_DIR = "/opt/nano"
NANO_BIN = f"{NANO_DIR}/bin/nano-claude"
# Order matters: testbed conda first so the agent's `python`/`pytest` resolve to
# the project's environment; nano-claude itself is invoked by absolute path.
AGENT_PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
# uv writes downloaded interpreters here (separate from its wheel cache); both
# are bind-mounted so they're shared across every container.
UV_CACHE = "/root/.cache/uv"
UV_PYTHON_DIR = "/root/.uv-python"

# Env vars forwarded into the container so litellm can reach the model provider.
# Forwarded by name only — values stay out of argv/logs. (See project memory:
# never echo API keys.)
_FORWARD_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "LITELLM_PROXY_API_KEY",
)


@dataclass
class Tooling:
    """Host-side artifacts shared by every container in a run."""

    wheel: Path  # nano-claude wheel (pure-python, arch-independent)
    uv_cache: Path  # host dir bind-mounted to UV_CACHE
    uv_python: Path  # host dir bind-mounted to UV_PYTHON_DIR


def prepare_tooling(work_dir: Path) -> Tooling:
    """Build the nano-claude wheel and create shared uv cache dirs (once)."""
    # Absolute paths: Docker treats a relative -v source as a *named volume*.
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    uv_cache = work_dir / "uv-cache"
    uv_python = work_dir / "uv-python"
    out = work_dir / "wheel"
    for d in (uv_cache, uv_python, out):
        d.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(PROJECT_ROOT)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    wheels = sorted(out.glob("nano_claude-*.whl"))
    if not wheels:
        raise RuntimeError("uv build produced no nano_claude wheel")
    return Tooling(wheel=wheels[-1], uv_cache=uv_cache, uv_python=uv_python)


def present_key_names(environ: dict[str, str] | None = None) -> list[str]:
    """Names (not values) of the forwarded API-key vars that are set."""
    environ = os.environ if environ is None else environ
    return [k for k in _FORWARD_ENV_KEYS if environ.get(k)]


def agent_argv(cfg: RolloutConfig) -> list[str]:
    """The in-container nano-claude command (one-shot, no prompting)."""
    return [
        NANO_BIN,
        "--stdin",
        "--model",
        cfg.model,
        "--max-turns",
        str(cfg.max_turns),
        "--permission-mode",
        "bypassPermissions",
    ]


def _test_cmd(task: Task) -> str:
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

    spec = MAP_REPO_VERSION_TO_SPECS.get(task.repo, {}).get(task.extra.get("version", ""), {})
    return str(spec.get("test_cmd") or "python -m pytest")


def _spec(task: Task):
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec(task.extra["instance"])


def prebuild_images(rows: list[dict], workers: int, log_dir: Path) -> set[str]:
    """Build base/env/instance images for every row; return built instance ids.

    Front-loaded before the rollout pool so the workers only create containers
    (no concurrent image builds racing on shared base/env layers).
    """
    import docker
    from swebench.harness.constants import LATEST
    from swebench.harness.docker_build import build_instance_images

    log_dir.mkdir(parents=True, exist_ok=True)
    client = docker.from_env()
    successful, failed = build_instance_images(
        client=client,
        dataset=rows,
        max_workers=max(1, workers),
        tag=LATEST,
        env_image_tag=LATEST,
    )

    def _iid(payload) -> str:
        # run_threadpool returns the *payloads* it was given; for
        # build_instance_image each payload is (test_spec, client, logger, nocache).
        spec = payload[0] if isinstance(payload, (tuple, list)) else payload
        return spec.instance_id

    return {_iid(s) for s in successful}


def _docker(
    args: list[str],
    *,
    stdin: str | None = None,
    timeout: int | None = None,
    capture: bool = True,
):
    """Run a `docker` CLI command in this process's environment.

    Env values for any `-e NAME` flags in `args` are taken from `os.environ`,
    so secrets are passed by reference and never appear in argv.
    """
    return subprocess.run(
        ["docker", *args],
        input=stdin,
        capture_output=capture,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )


def _exec(
    container: str,
    argv: list[str],
    *,
    workdir: str | None = None,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
    timeout: int | None = None,
):
    args = ["exec", "-i"]
    if workdir:
        args += ["-w", workdir]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    args += [container, *argv]
    return _docker(args, stdin=stdin, timeout=timeout)


def _setup_nano(container: str, tooling: Tooling) -> None:
    """Install nano-claude into /opt/nano (py3.12) inside the container."""
    uv_env = {"UV_CACHE_DIR": UV_CACHE, "UV_PYTHON_INSTALL_DIR": UV_PYTHON_DIR}
    # Bootstrap uv via the image's miniconda base interpreter.
    r = _exec(
        container, ["/opt/miniconda3/bin/python", "-m", "pip", "install", "-q", "uv"], timeout=600
    )
    _check(r, "pip install uv")
    # Copy the wheel in, create an isolated 3.12 venv, install nano-claude.
    dest = f"{container}:/tmp/{tooling.wheel.name}"
    _check(_docker(["cp", str(tooling.wheel), dest], timeout=120), "docker cp wheel")
    r = _exec(
        container,
        ["/opt/miniconda3/bin/uv", "venv", "--python", "3.12", NANO_DIR],
        env=uv_env,
        timeout=900,
    )
    _check(r, "uv venv")
    r = _exec(
        container,
        [
            "/opt/miniconda3/bin/uv",
            "pip",
            "install",
            "--python",
            f"{NANO_DIR}/bin/python",
            f"/tmp/{tooling.wheel.name}",
        ],
        env=uv_env,
        timeout=1800,
    )
    _check(r, "uv pip install nano-claude")


def _seed_exclude(container: str) -> None:
    """Keep build artifacts / scratch DBs out of the captured patch."""
    block = "\n# nano-eval\n" + "\n".join(_LOCAL_EXCLUDE) + "\n"
    _exec(
        container,
        ["bash", "-c", f"cat >> {TESTBED}/.git/info/exclude"],
        stdin=block,
        timeout=60,
    )


def _capture_patch(container: str, task: Task) -> str:
    r = _exec(
        container,
        ["bash", "-c", f"cd {TESTBED} && git add -A && git diff --cached {task.base_commit}"],
        timeout=120,
    )
    _check(r, "capture patch")
    return r.stdout


def _check(r: subprocess.CompletedProcess, what: str) -> None:
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[-800:]
        raise RuntimeError(f"{what} failed (exit {r.returncode}): {err}")


def run_task_docker(
    task: Task,
    cfg: RolloutConfig,
    log_dir: Path,
    run_id: str,
    tooling: Tooling,
) -> RolloutResult:
    """Run the agent inside the instance container and capture its diff."""
    import docker

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.instance_id}.log"
    started = time.monotonic()

    def elapsed() -> float:
        return time.monotonic() - started

    def fail(msg: str) -> RolloutResult:
        return RolloutResult(
            task.instance_id,
            task.repo,
            task.base_commit,
            RolloutStatus.ERROR,
            error=msg,
            duration_s=elapsed(),
            log_path=str(log_path),
            env_ready=False,
        )

    client = docker.from_env()
    spec = _spec(task)
    image = spec.instance_image_key
    try:
        client.images.get(image)
    except Exception:  # noqa: BLE001 - image missing -> nothing to roll out
        return fail(f"instance image {image} not present (build failed?)")

    name = f"nano.rollout.{task.instance_id.lower()}.{run_id}"
    # Clear any leftover container with this name from a previous run.
    _docker(["rm", "-f", name], capture=True)

    container = None
    status = RolloutStatus.OK
    error: str | None = None
    try:
        container = client.containers.create(
            image=image,
            name=name,
            command="tail -f /dev/null",
            detach=True,
            platform=spec.platform,
            user="root",
            volumes={
                str(tooling.uv_cache): {"bind": UV_CACHE, "mode": "rw"},
                str(tooling.uv_python): {"bind": UV_PYTHON_DIR, "mode": "rw"},
            },
        )
        container.start()
        _setup_nano(name, tooling)
        _seed_exclude(name)

        prompt = task.prompt + verify_addendum(_test_cmd(task))
        with log_path.open("w") as log:
            agent = subprocess.Popen(
                [
                    "docker",
                    "exec",
                    "-i",
                    "-w",
                    TESTBED,
                    "-e",
                    f"PATH={AGENT_PATH}",
                    "-e",
                    "NANO_CLAUDE_DISABLE_MEMORY=1",
                    *sum((["-e", k] for k in present_key_names()), []),
                    name,
                    *agent_argv(cfg),
                ],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ},
            )
            try:
                agent.communicate(input=prompt, timeout=cfg.task_timeout)
                if agent.returncode != 0:
                    status = RolloutStatus.ERROR
                    error = f"nano-claude exited with code {agent.returncode}"
            except subprocess.TimeoutExpired:
                agent.kill()
                status = RolloutStatus.TIMEOUT
                error = f"agent exceeded {cfg.task_timeout}s"

        patch = ""
        try:
            patch = _capture_patch(name, task)
            if cfg.strip_test_changes:
                patch = strip_paths(patch, _test_paths(task))
        except Exception as exc:  # noqa: BLE001
            if status is RolloutStatus.OK:
                status = RolloutStatus.ERROR
                error = f"patch capture failed: {exc}"

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
            env_ready=True,
        )
    except Exception as exc:  # noqa: BLE001 - any setup failure ends the task
        return fail(str(exc))
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


def _test_paths(task: Task) -> set[str]:
    test_patch = task.extra.get("test_patch", "")
    return changed_paths(test_patch) if test_patch else set()
