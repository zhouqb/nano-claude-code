# nano-claude-code

A minimal but functional implementation of Claude Code in Python — a streaming
coding agent with tools, a permission system, session persistence, a five-layer
context-compaction pipeline, an extensibility layer (hooks, skills, MCP servers,
plugins), and single-level subagents. See
[`nano-claude-code-plan.md`](./nano-claude-code-plan.md) for the full design.

Multi-provider model access is via [LiteLLM](https://github.com/BerriAI/litellm),
so any provider it supports works through `--model`.

## Features

- **Streaming agent loop** — concurrent tool dispatch, retry with backoff,
  cancellation.
- **Tools** — `Bash`, `Read`, `Write`, `Edit`, `GlobTool`, `Grep`, plus `Task`
  (subagent delegation) and any tools contributed by MCP servers/plugins.
- **Permissions** — three modes (`default` / `acceptEdits` / `bypassPermissions`)
  with allow/deny rules (`Bash(git *)` grammar) persisted to settings.
- **Sessions** — every turn is written to JSONL; `--resume` reopens a session
  and repairs a mid-turn crash.
- **Context compaction** — a fixed five-layer pipeline runs before each request:
  budget reduction → snip → microcompact → context collapse → auto-compact.
- **Extensibility** — hooks, `/command` skills, MCP servers, and plugins, all
  wired at startup.
- **Subagents** — delegate a noisy exploration to an isolated agent via `Task`;
  only its final summary returns to the parent.

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Usage

Set the API key for whichever provider you target (`DEEPSEEK_API_KEY` for the
default model, or e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …), then:

```bash
nano-claude                                   # defaults to deepseek/deepseek-chat
nano-claude --model gpt-4o
nano-claude --model anthropic/claude-sonnet-4-6 --max-turns 20
nano-claude --resume                          # pick a previous session to continue
nano-claude --permission-mode acceptEdits     # auto-allow file edits, still prompt for Bash
```

The model can also be set via `NANO_CLAUDE_MODEL`.

### REPL commands

| Command | Action |
|---|---|
| `/help` | List commands, skills, and agents. |
| `/cost` | Token usage and estimated cost for this session. |
| `/model` | Current model and context window. |
| `/compact` | Summarize the conversation to free context. |
| `/clear` | Start a fresh session. |
| `/init` | Analyze the codebase and draft a `CLAUDE.md`. |
| `/quit` | Exit (or Ctrl-D). |

## Extending

Configuration lives under `~/.nano-claude/`:

```
~/.nano-claude/
  settings.json     # permission mode + rules, "hooks", "mcpServers"
  skills/           # <name>.md (frontmatter) or <name>.py  → /command shortcuts
  agents/           # <name>.md (frontmatter)               → subagents for Task
  plugins/          # <plugin>/manifest.json bundling the above
```

**Hooks** — shell commands fired at lifecycle events. `PreToolUse` can block a
call (exit code 2); `PostToolUse` stdout is appended to the tool result;
`SessionStart`/`Stop` run at the edges. The hook receives a JSON payload on
stdin.

```json
{
  "hooks": [
    {"event": "PreToolUse",  "matcher": "Bash(rm *)", "command": "./guard.sh"},
    {"event": "PostToolUse", "matcher": "Edit",       "command": "ruff format \"$NANO_CLAUDE_TOOL\""}
  ]
}
```

**Skills** — a `/command` that expands into a prompt. Markdown form:

```markdown
---
name: commit
description: Stage changes and write a commit message
argument-hint: "[scope]"
allowed-tools: [Bash, Read]
---
Review the staged diff and create a git commit. Scope: $ARGUMENTS
```

**MCP servers** — external tool providers, surfaced as `mcp__<server>__<tool>`:

```json
{
  "mcpServers": {
    "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}
  }
}
```

**Agents** — subagents the model can delegate to via `Task`:

```markdown
---
name: explorer
description: Read-only codebase search. Use for broad fan-out searches.
tools: [Read, Grep, GlobTool]
model: anthropic/claude-haiku-4-5
---
You are a read-only exploration agent. Report findings as a short summary with
file:line references.
```

**Plugins** bundle hooks, skills, and MCP servers behind one `manifest.json`.

## Development

```bash
.venv/bin/ruff check .        # lint
.venv/bin/ruff format .       # format
.venv/bin/pytest              # tests
```
