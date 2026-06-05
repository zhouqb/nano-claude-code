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
- Do NOT modify, add, or delete any tests. The grader supplies its own tests; \
changing test files will invalidate the run.
- You may read and search the code freely. You generally cannot run the test \
suite (dependencies are not installed), so reason carefully about correctness.
- Do not commit, and do not run `git` write commands. Just leave your edits in \
the working tree when you are done.

--- ISSUE ---
{problem_statement}
"""


def swe_bench_prompt(repo: str, problem_statement: str) -> str:
    return SWE_BENCH_PROMPT.format(repo=repo, problem_statement=problem_statement)
