"""Agent discovery and the global agent registry.

Agents are loaded once at startup from ``~/.nano-claude/agents/*.md`` (Markdown
with YAML frontmatter), plus a built-in ``general-purpose`` agent registered in
code so delegation works out of the box.
"""

from __future__ import annotations

from pathlib import Path

import frontmatter

from nano_claude.subagents.types import AgentDefinition

DEFAULT_AGENTS_DIR = Path.home() / ".nano-claude" / "agents"

# Populated at startup by load_agents().
AGENT_REGISTRY: dict[str, AgentDefinition] = {}

# Always available, even with no agent files on disk.
GENERAL_PURPOSE = AgentDefinition(
    name="general-purpose",
    description=(
        "General-purpose agent for researching complex questions, searching code, "
        "and executing multi-step tasks. Use when a task needs several rounds of "
        "exploration you don't want filling the main context."
    ),
    system_prompt=(
        "You are a general-purpose subagent. Complete the delegated task "
        "autonomously using the tools available to you, then report back a "
        "concise, self-contained summary of what you found or did, including "
        "any file:line references the caller will need. You cannot ask the "
        "caller follow-up questions, so make reasonable assumptions and state "
        "them."
    ),
)

_VERIFICATION_SYSTEM_PROMPT = (
    "You are a verification specialist. Your job is not to confirm the "
    "implementation works — it is to try to break it.\n\n"
    "You have two documented failure patterns. First, verification avoidance: "
    "faced with a check, you find reasons not to run it — you read the code, "
    'narrate what you would test, write "PASS," and move on. Second, being '
    "seduced by the first 80%: a passing test suite or a plausible diff makes you "
    "want to pass it, while the actual bug is untouched. The implementer is an LLM "
    "too; its tests may be heavy on mocks, circular assertions, or happy-path "
    "coverage that proves nothing. Run things; don't read and assume.\n\n"
    "DO NOT MODIFY THE PROJECT. You must not create, edit, or delete files in the "
    "project directory, install packages, or run git write commands. You MAY write "
    "ephemeral scratch scripts under /tmp (or $TMPDIR) and run them; clean up after "
    "yourself.\n\n"
    "You receive: the original task, the files changed, and the approach taken.\n\n"
    "Required baseline (in order):\n"
    "1. Read CLAUDE.md / README / pyproject.toml / package.json / Makefile to learn "
    "the real build and test commands and conventions. Do not assume them.\n"
    "2. Run the build if there is one. A broken build is an automatic FAIL.\n"
    "3. Run the relevant tests. Failing tests are an automatic FAIL.\n"
    "4. Run linters / type-checkers if configured.\n"
    "5. Check for regressions in code related to the change.\n\n"
    "Then verify the change directly. For a bug fix: reproduce the original bug "
    "from the issue description FIRST (confirm it actually misbehaves), then run "
    "the same scenario against the fixed code and confirm the symptom is gone, then "
    "run regression tests. Anchor your reproduction on the reported symptom, not on "
    "the file the implementer happened to edit — that is how you catch a fix made "
    "in the wrong place. Before issuing PASS, run at least one adversarial probe "
    "(boundary value, empty/oversized input, idempotency, unexpected type) and "
    "report its result.\n\n"
    'Recognize your own rationalizations and do the opposite: "the code looks '
    'correct" (reading is not verification — run it); "the implementer\'s tests '
    'pass" (verify independently); "this is probably fine" (probably is not '
    "verified). If you catch yourself writing an explanation instead of a command, "
    "stop and run the command.\n\n"
    "Output: for every check, show the exact command you ran and its actual output "
    "(copy-paste, not paraphrased), then Result: PASS or FAIL (with Expected vs "
    "Actual). A check with no command run is a skip, not a PASS. End your report "
    "with exactly one final line: the literal string 'VERDICT: ' followed by PASS, "
    "FAIL, or PARTIAL. Use PARTIAL only for environmental limits (no test "
    'framework, a tool is unavailable, the server cannot start) — not for "I am '
    'unsure." If you can run the check, decide PASS or FAIL.'
)

# Adversarial, read-only verifier spawned before reporting non-trivial work done.
VERIFICATION = AgentDefinition(
    name="verification",
    description=(
        "Adversarial verification specialist. Spawn it after non-trivial "
        "implementation (roughly 3+ files changed, or any backend/API, data-model, "
        "or infrastructure change) to independently verify the work BEFORE you "
        "report completion. Pass the original task, the files changed, and the "
        "approach taken. It runs builds, tests, and adversarial probes read-only "
        "(it cannot edit the project) and returns a PASS/FAIL/PARTIAL verdict "
        "backed by command output."
    ),
    system_prompt=_VERIFICATION_SYSTEM_PROMPT,
    # Read-only: no Write/Edit/TodoWrite. Bash is allowed (tests, /tmp scratch);
    # the prompt forbids modifying the project, mirroring Claude Code's verifier.
    tools=["Bash", "Read", "Grep", "GlobTool"],
)


def register_agent(agent: AgentDefinition) -> None:
    AGENT_REGISTRY[agent.name] = agent


def clear_agents() -> None:
    AGENT_REGISTRY.clear()


def get_agent(name: str) -> AgentDefinition | None:
    return AGENT_REGISTRY.get(name)


def _parse_tools(raw: object) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _agent_from_markdown(path: Path) -> AgentDefinition:
    post = frontmatter.load(str(path))
    meta = post.metadata
    return AgentDefinition(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description", "")),
        system_prompt=post.content.strip(),
        tools=_parse_tools(meta.get("tools")),
        model=(str(meta["model"]) if meta.get("model") else None),
    )


def load_agents(directory: Path | None = None) -> list[AgentDefinition]:
    """Register the built-in agents plus every agent file in ``directory``."""
    register_agent(GENERAL_PURPOSE)
    register_agent(VERIFICATION)

    directory = directory or DEFAULT_AGENTS_DIR
    loaded: list[AgentDefinition] = [GENERAL_PURPOSE, VERIFICATION]
    if not directory.is_dir():
        return loaded

    for path in sorted(directory.iterdir()):
        if path.suffix != ".md":
            continue
        try:
            agent = _agent_from_markdown(path)
        except Exception:  # noqa: BLE001 - one bad agent must not break startup
            continue
        register_agent(agent)
        loaded.append(agent)
    return loaded
