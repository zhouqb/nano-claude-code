"""The memory section injected into the system prompt.

Teaches the model the four-type taxonomy, what *not* to save, the two-step save
flow, and how to treat recalled facts — then appends the current ``MEMORY.md``
index (always loaded, truncated to the caps). Condensed from
``src/memdir/memoryTypes.ts`` + ``memdir.ts`` in the Claude Code source.
"""

from __future__ import annotations

from pathlib import Path

from nano_claude.memory.paths import ENTRYPOINT
from nano_claude.memory.store import ensure_memory_dir, read_entrypoint, truncate_entrypoint

MEMORY_TYPES = ("user", "feedback", "project", "reference")

_TAXONOMY = """\
## Types of memory

Every memory is exactly one of these four types. Anything derivable from the
current project state is NOT a memory (see "What NOT to save").

<types>
<type>
  <name>user</name>
  Who the user is — role, expertise, preferences. Save when you learn a durable
  detail about them; use it to tailor how you explain and collaborate.
</type>
<type>
  <name>feedback</name>
  Guidance on how to work — both corrections ("no, don't do X") AND confirmations
  ("yes, keep doing that"). Save the rule, then a **Why:** line (the reason, often
  a past incident) and a **How to apply:** line (when it kicks in). Watch for
  quiet confirmations, not just corrections.
</type>
<type>
  <name>project</name>
  Ongoing work, goals, decisions, incidents not derivable from code or git. Save
  who is doing what, why, and by when; **convert relative dates to absolute**
  (e.g. "Thursday" → an ISO date). Lead with the fact, then **Why:** / **How to apply:**.
</type>
<type>
  <name>reference</name>
  Pointers to external systems (a Linear project, a Grafana dashboard, a Slack
  channel) and what they are for.
</type>
</types>"""

_WHAT_NOT_TO_SAVE = """\
## What NOT to save

- Code patterns, architecture, file paths, project structure — derivable by reading the project.
- Git history or who-changed-what — `git log` / `git blame` are authoritative.
- Debugging fix recipes — the fix is in the code; the commit message has the context.
- Anything already in CLAUDE.md, or ephemeral in-progress / current-conversation state.

These exclusions hold even when the user asks you to save. If they ask you to
save an activity summary, ask what was *surprising* or *non-obvious* — that is
the part worth keeping."""

_ACCESS_AND_RECALL = """\
## When to access and how to trust memory

- Consult memory when it seems relevant or the user references prior-conversation work; you MUST when they ask you to recall or remember.
- If the user says to *ignore* memory, behave as if it were empty — don't cite, compare against, or apply it.
- A memory is a point-in-time snapshot. Before recommending a file, function, or flag it names, verify it still exists (read the file / grep). "The memory says X exists" is not "X exists now"."""


def _how_to_save(mdir: Path) -> str:
    return f"""\
## How to save memories

Saving is two steps:
1. Write the memory to its own file (e.g. `user_role.md`) with this frontmatter:
```markdown
---
name: {{short-name}}
description: {{one-line summary — used to judge relevance during recall, so be specific}}
type: {{{" | ".join(MEMORY_TYPES)}}}
---
{{the fact; for feedback/project add **Why:** and **How to apply:** lines}}
```
2. Add a one-line pointer to `{ENTRYPOINT}`: `- [Title](file.md) — one-line hook`.
   `{ENTRYPOINT}` is an always-loaded index, not a memory — never write content into it.

- Before creating a file, check for an existing one to update — do not duplicate.
- Organize by topic, not chronologically. Delete memories that turn out wrong.
- Your memory directory is `{mdir}`. It already exists — write to it directly."""


def build_memory_section(mdir: Path) -> str:
    """Build the full memory system-prompt block, including the index content."""
    ensure_memory_dir(mdir)
    parts = [
        "# Memory",
        "",
        "You have a persistent, file-based memory that survives across sessions. "
        "Build it up over time so future conversations understand the user, how to "
        "work with them, and the context behind their requests.",
        "",
        _TAXONOMY,
        "",
        _WHAT_NOT_TO_SAVE,
        "",
        _how_to_save(mdir),
        "",
        _ACCESS_AND_RECALL,
        "",
    ]

    raw = read_entrypoint(mdir)
    if raw.strip():
        content, _ = truncate_entrypoint(raw)
        parts.append(f"## {ENTRYPOINT}\n\n{content}")
    else:
        parts.append(
            f"## {ENTRYPOINT}\n\nYour {ENTRYPOINT} is empty. Saved memories will appear here."
        )
    return "\n".join(parts)
