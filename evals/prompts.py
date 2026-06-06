"""Prompt templates for datasets that don't ship a ready-to-use agent prompt.

SWE-bench instances provide a raw GitHub ``problem_statement`` but no agent
instructions, so we wrap it once here. Keep the guidance terse and behavioral;
nano-claude already knows how to use its tools.
"""

from __future__ import annotations

SWE_BENCH_PROMPT = """\
You are working at the root of the `{repo}` repository, checked out at the commit \
where the issue below was reported. Resolve the issue by editing the repository's \
source code in place.

Guidelines:
- Make the smallest change that correctly fixes the issue.
- Do NOT modify, add, or delete the project's own test files. The grader applies its \
own tests on top of the repository's existing test suite, and your change must not \
break tests that currently pass; editing test files will invalidate the run. (You may \
still write a scratch reproduction script outside the test suite to check your work.)
- You may read and search the code freely.
- Do not commit, and do not run `git` write commands. Just leave your edits in \
the working tree when you are done.

--- ISSUE ---
{problem_statement}
"""


def swe_bench_prompt(repo: str, problem_statement: str) -> str:
    return SWE_BENCH_PROMPT.format(repo=repo, problem_statement=problem_statement)


def verify_addendum(test_cmd: str) -> str:
    """Appended to the prompt when a working test environment is available."""
    return (
        "\n\n--- VERIFICATION ---\n"
        "The project and its dependencies are installed in this environment, so you "
        "CAN run the test suite to verify your change. Run the tests you judge "
        f"relevant to the issue, e.g. `{test_cmd} <path-or-test-ids>` or "
        "`python -m pytest <path>`. Don't just reason about correctness — run tests, "
        "read the failures, and iterate until the relevant tests pass. Avoid running "
        "the entire suite (slow); target the modules related to the issue. "
        "Any test that passes on the unmodified code must still pass after your "
        "change: if a test fails once you've edited, assume your change caused it and "
        "fix your change — do not dismiss failures as pre-existing (check against the "
        "original code first) or assume the grader's own tests will replace them."
    )
