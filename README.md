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
- **One-shot CLI mode** — pass a prompt on the command line or via `--stdin`
  to run exactly one turn and exit.
- **Context compaction** — a fixed five-layer pipeline runs before each request:
  budget reduction → snip → microcompact → context collapse → auto-compact.
- **Extensibility** — hooks, `/command` skills, MCP servers, and plugins, all
  wired at startup.
- **Subagents** — delegate a noisy exploration to an isolated agent via `Task`;
  only its final summary returns to the parent.
- **Observability** — opt-in OpenTelemetry traces (OTLP) and per-session logs.

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
nano-claude                                   # defaults to deepseek/deepseek-v4-flash
nano-claude --model gpt-4o
nano-claude --model anthropic/claude-sonnet-4-6 --max-turns 20
nano-claude --resume                          # pick a previous session to continue
nano-claude --permission-mode acceptEdits     # auto-allow file edits, still prompt for Bash
nano-claude "summarize this repository"       # single-turn run, then exit
echo "review the staged diff" | nano-claude --stdin
```

For one-shot runs, if you do not explicitly pass `--permission-mode`, nano
defaults that invocation to `bypassPermissions`. Pass `--permission-mode
default` or `--permission-mode acceptEdits` to override that behavior.

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

## Observability

Telemetry is opt-in and off by default. With the `otel` extra installed (already
in `[dev]`, otherwise `uv pip install -e ".[otel]"`), set `NANO_CLAUDE_TELEMETRY=1`
to emit OpenTelemetry traces over OTLP/HTTP — `agent.turn` → `chat <model>` (with
GenAI usage attributes) → `tool <name>` spans:

```bash
NANO_CLAUDE_TELEMETRY=1 nano-claude
```

Spans export to `http://localhost:4318` by default; override with the standard
`OTEL_EXPORTER_OTLP_ENDPOINT`. Point them at any OTLP backend — the quickest way
to get a trace viewer is [Jaeger](https://www.jaegertracing.io/) all-in-one,
which ingests OTLP natively and serves a UI on port 16686:

```bash
docker run --rm --name jaeger -p 16686:16686 -p 4318:4318 -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

No Docker? Grab the [Jaeger release binary](https://github.com/jaegertracing/jaeger/releases)
for your platform and run `./jaeger` — it opens the same ports.

### Viewing traces

1. Start a backend (the Jaeger command above) and leave it running.
2. Run nano-claude with telemetry on and have a conversation — tool-using turns
   produce the richest spans:

   ```bash
   NANO_CLAUDE_TELEMETRY=1 nano-claude
   ```

3. Open the Jaeger UI at **http://localhost:16686**, pick the `nano-claude-code`
   service in the *Service* dropdown, and click **Find Traces**. Each `agent.turn`
   expands into its `chat <model>` and `tool <name>` children. Click **Find
   Traces** again to refresh after later turns.

Each `chat` span carries the full exchange — prompt messages plus the assistant
reply appended as the last message — in a single `gen_ai.messages` field; each
`tool` span carries its arguments, output, and error
(`nano_claude.tool.arguments` / `.output` / `.error`). These payloads can be
large or sensitive — set `NANO_CLAUDE_TELEMETRY_CAPTURE_CONTENT=0` to drop them,
or `NANO_CLAUDE_TELEMETRY_MAX_CONTENT_LEN` to change the truncation cap (default
16384 chars, applied per string value so the serialized JSON stays valid).

Spans batch and flush at exit, so quit nano-claude cleanly (`/quit` or Ctrl-D) if
a trace hasn't appeared yet, then refresh.

Logs are a separate signal: by default they go to a per-session file
(`~/.nano-claude/projects/<cwd>/<session-id>.log.jsonl`), not the OTLP backend.
See [`CLAUDE.md`](./CLAUDE.md) for the full set of `NANO_CLAUDE_TELEMETRY_*` knobs
(console exporters, logs-over-OTLP, etc.).

## Development

```bash
.venv/bin/ruff check .        # lint
.venv/bin/ruff format .       # format
.venv/bin/pytest              # tests
```
