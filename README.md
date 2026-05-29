# nano-claude-code

A minimal viable implementation of Claude Code in Python. See
[`nano-claude-code-plan.md`](./nano-claude-code-plan.md) for the full design.

## Status

**Phase 1 — Skeleton** ✅

- `click` CLI with `--model`, `--max-turns`, `--permission-mode`
- Streaming agent loop (single model call per turn; tools land in Phase 2)
- System prompt assembly (OS, shell, cwd, git branch, date, CLAUDE.md)
- Interactive REPL with streamed output
- Smoke tests for the loop and context builders

Multi-provider model access is via [LiteLLM](https://github.com/BerriAI/litellm),
so any provider it supports works through `--model`.

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Usage

Set the API key for whichever provider you target (e.g. `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, …), then:

```bash
nano-claude                                   # defaults to anthropic/claude-sonnet-4-6
nano-claude --model gpt-4o
nano-claude --model deepseek/deepseek-chat --max-turns 20
```

The model can also be set via `NANO_CLAUDE_MODEL`. Type `/quit` (or Ctrl-D) to exit.

## Development

```bash
.venv/bin/ruff check .        # lint
.venv/bin/ruff format .       # format
.venv/bin/pytest              # tests
```
