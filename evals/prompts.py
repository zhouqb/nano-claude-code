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
- Make the smallest change that COMPLETELY fixes the issue — fix the root cause, \
not just the one example, but don't gold-plate.
- Your deliverable is a NON-EMPTY change to the project's SOURCE code (non-test \
files). Tests verify the fix; they are not the fix. Land a working source fix \
first — don't spend more than a turn or two writing tests before you have one — \
and never finish with only test changes. You are still encouraged to add or change \
tests to reproduce and verify the bug (the project's own framework is fine), but \
the grader reverts ALL of your test-file changes before grading and runs its own \
suite, so test edits never count for or against you and cannot be used to make the \
suite pass — only your source change is graded, and it must not break tests that \
currently pass.
- You may read and search the code freely.
- Do not commit, and do not run `git` write commands. Just leave your edits in \
the working tree when you are done.

--- ISSUE ---
{problem_statement}
"""


def swe_bench_prompt(repo: str, problem_statement: str) -> str:
    return SWE_BENCH_PROMPT.format(repo=repo, problem_statement=problem_statement)


VERIFICATION_PASS_PROMPT = """\
You are an INDEPENDENT verifier. Another agent has already edited the `{repo}` \
working tree to resolve the issue below; its changes are live in the tree right \
now (run `git diff` to see them). You did not write them, so assume they are \
WRONG until you prove otherwise — your job is to catch what the implementer, \
invested in its own fix, talked itself past.

Verify, then repair only if needed:
1. REPRODUCE: from the issue, construct the exact case it reports and run it on \
the CURRENT tree. Confirm the behavior is genuinely correct now — not merely \
different, non-crashing, or "looks right".
2. HUNT REGRESSIONS (most important): run the existing tests that exercise the \
code that was touched — search the test tree for the changed functions/classes \
and run those modules with `{test_cmd} <path-or-ids>` (or `python -m pytest \
<path>`). Any test that would have passed on the original code but fails now is \
a regression the fix introduced; it MUST be repaired.
3. ATTACK THE EDGES: try to break the fix — other operators/types, empty / None \
/ zero / negative / infinity, symmetric and inverse paths, and the wider family \
the issue implies beyond its one literal example. A fix that only handles the \
spelled-out example is incomplete.

If everything holds, change nothing and say so. If you find a real failing case \
or regression, fix it IN THE SOURCE (non-test files) — make the smallest change \
that addresses the root cause. Hard constraints:
- Do NOT discard or empty out the existing fix. A correct-but-incomplete fix \
must be EXTENDED, never reverted to nothing. If you are unsure, leave the \
existing source change in place.
- Your test edits are reverted before grading, so they never count — only the \
source change is graded, and it must not break tests that currently pass.
- Do not commit or run `git` write commands; leave edits in the working tree.

End by stating your VERDICT (PASS / PARTIAL-now-fixed / still-FAILING), the \
cases you ran, and exactly what you changed, if anything.

--- ISSUE ---
{problem_statement}
"""


def verification_pass_prompt(repo: str, problem_statement: str, test_cmd: str) -> str:
    """User prompt for the enforced, independent verification pass (eval rollout).

    Unlike the read-only ``verification`` subagent (which only reports a
    verdict), this pass may repair what it finds, because the offline rollout
    has no human in the loop to act on a verdict.
    """
    return VERIFICATION_PASS_PROMPT.format(
        repo=repo, problem_statement=problem_statement, test_cmd=test_cmd
    )


def verify_addendum(test_cmd: str) -> str:
    """Appended to the prompt when a working test environment is available."""
    return (
        "\n\n--- VERIFICATION ---\n"
        "The project and its dependencies are installed here, so you can run code and "
        "tests. Verify by REPRODUCING the issue first, then fixing — never the reverse:\n"
        "1. UNDERSTAND the issue, then write a SMALL test that reproduces it — a real "
        "test in the project's own framework (for its fixtures, settings, and test "
        "types like image comparison) or a quick script in /tmp. Keep this lightweight: "
        "your goal is a working source fix, not an elaborate test suite, so don't sink "
        "your budget here.\n"
        "2. RUN that test on the UNMODIFIED code and confirm it FAILS — paste the "
        "failing output. This is a hard gate: if it does not fail, you have NOT "
        "reproduced the real bug. Do not edit any source until you have a failing "
        "reproduction. The existing suite already passes on the broken code (that is "
        "why the bug shipped), so 'the suite is green' proves nothing — the proof is "
        "your reproduction going from fail to pass.\n"
        "3. Fix the ROOT CAUSE, not the one trigger. Read where the project already "
        "tests the code you're changing (search the test tree for the function/class/"
        "symbol) to learn the contract and edge cases. If you find an invariant being "
        "violated (e.g. a method returning a wrong or negative value) AND a specific "
        "operation that triggers it, fix the invariant itself so EVERY path is covered "
        "— not just the operation in your reproduction. Cover the whole family the "
        "issue implies (other operators, types, boundary/empty/None/infinity values, "
        "symmetric paths). Every failing case your reproduction surfaced must pass "
        "after the fix — do not fix one symptom and leave another you already saw.\n"
        f"4. Re-run your reproduction — it MUST now pass; that is your final proof, not "
        f"'the existing suite is green'. If you can't point to a specific case that went "
        "from fail to pass because of your edit, you are not done. Then run the relevant "
        f"existing tests (`{test_cmd} <path-or-test-ids>` or `python -m pytest <path>`; "
        "target the related modules, the full suite is slow): any test that passed on "
        "the original code must still pass — if one fails after your edit, assume your "
        "change caused it and fix it; don't dismiss it as pre-existing or assume the "
        "grader's tests will replace it.\n"
        "5. If the behavior genuinely cannot be exercised here, say so and reason "
        "carefully about correctness rather than claiming success from a passing subset."
    )
