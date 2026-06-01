"""CLI entry point and interactive REPL.

Phase 3: conversations are persisted to JSONL as they happen, and ``--resume``
reopens a previous session (repaired if it was interrupted mid-turn).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.table import Table

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason, TokenUsage
from nano_claude.commands import format_cost, format_help, format_model, format_turn_footer
from nano_claude.compaction.compactor import compact_conversation
from nano_claude.context import build_system_prompt
from nano_claude.extensibility.hooks import HookEvent, execute_hooks
from nano_claude.extensibility.loader import load_extensions
from nano_claude.extensibility.mcp import close_mcp
from nano_claude.extensibility.skills import SKILL_REGISTRY, SkillContext, dispatch_skill
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.prompt import make_cli_prompter
from nano_claude.permissions.settings import Settings
from nano_claude.session.restore import (
    last_assistant_ts,
    list_sessions,
    load_records,
    repair_messages,
    restore_messages,
    restore_read_file_state,
)
from nano_claude.session.storage import SessionStorage, new_session_id, session_file
from nano_claude.subagents import AGENT_REGISTRY, load_agents
from nano_claude.ui import ReplUI

DEFAULT_MODEL = os.environ.get("NANO_CLAUDE_MODEL", "anthropic/claude-sonnet-4-6")

console = Console()

INIT_PROMPT = """Please analyze this codebase and create a CLAUDE.md file, which will be given to future instances of Claude Code to operate in this repository.

What to add:
1. Commands that will be commonly used, such as how to build, lint, and run tests. Include the necessary commands to develop in this codebase, such as how to run a single test.
2. High-level code architecture and structure so that future instances can be productive more quickly. Focus on the "big picture" architecture that requires reading multiple files to understand.

Usage notes:
- If there's already a CLAUDE.md, suggest improvements to it.
- When you make the initial CLAUDE.md, do not repeat yourself and do not include obvious instructions like "Provide helpful error messages to users", "Write unit tests for all new utilities", "Never include sensitive information (API keys, tokens) in code or commits".
- Avoid listing every component or file structure that can be easily discovered.
- Don't include generic development practices.
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include the important parts.
- If there is a README.md, make sure to include the important parts.
- Do not make up information such as "Common Development Tasks", "Tips for Development", "Support and Documentation" unless this is expressly included in other files that you read.
- Be sure to prefix the file with the following text:

```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```"""


def _resolve_context_window(model: str, fallback: int) -> int:
    """Look up the model's context window via LiteLLM, falling back on failure."""
    try:
        import litellm

        info = litellm.get_model_info(model)
    except Exception:
        return fallback
    window = (info or {}).get("max_input_tokens") or (info or {}).get("max_tokens")
    return int(window) if window else fallback


def _pick_session(cwd: str):
    """Show recent sessions for cwd and let the user choose one to resume."""
    sessions = list_sessions(cwd)
    if not sessions:
        console.print("[yellow]No previous sessions found; starting a new one.[/yellow]")
        return None

    table = Table(title="Resumable sessions", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("When")
    table.add_column("Model")
    table.add_column("First message")
    for i, s in enumerate(sessions[:20], start=1):
        when = datetime.fromtimestamp(s.mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(str(i), when, s.model, s.preview)
    console.print(table)

    raw = input("Select a session number (Enter to start new): ").strip()
    if not raw:
        return None
    try:
        idx = int(raw) - 1
    except ValueError:
        console.print("[yellow]Not a number; starting a new session.[/yellow]")
        return None
    if 0 <= idx < len(sessions):
        return sessions[idx]
    console.print("[yellow]Out of range; starting a new session.[/yellow]")
    return None


async def _fire_session_start(config: AgentConfig, state: LoopState) -> None:
    """Run SessionStart hooks; inject their stdout as a context note for the model."""
    session_id = state.storage.session_id if state.storage is not None else ""
    outcome = await execute_hooks(HookEvent.SESSION_START, session_id=session_id, cwd=config.cwd)
    for warning in outcome.warnings:
        console.print(f"[yellow]SessionStart hook: {warning}[/yellow]")
    if outcome.context_text:
        note = {"role": "system", "content": f"[SessionStart hook]\n{outcome.context_text}"}
        state.messages.append(note)
        if state.storage is not None:
            state.storage.append_message(note)


async def _repl(config: AgentConfig, settings: Settings, state: LoopState) -> None:
    session: PromptSession = PromptSession()
    prompter = make_cli_prompter(session)
    storage = state.storage
    ui = ReplUI(console)

    console.print(
        f"[bold cyan]nano-claude-code[/bold cyan] [dim]({config.model}, "
        f"mode={config.permission_mode.value}, session={storage.session_id})[/dim]"
    )
    console.print("[dim]Type your message. /help for commands, /quit or Ctrl-D to exit.[/dim]\n")

    # Wire all extensions inside the event loop (same task as close_mcp, so the
    # MCP SDK's cancel scopes stay in one task): hooks, skills, MCP, plugins.
    summary = await load_extensions(settings)
    if summary.anything:
        console.print(
            f"[dim]Extensions: {summary.hooks} hook(s), {summary.skills} skill(s), "
            f"{summary.mcp_tools} MCP tool(s), {summary.plugins} plugin(s).[/dim]"
        )

    await _fire_session_start(config, state)

    try:
        while True:
            try:
                with patch_stdout():
                    user_input = await session.prompt_async("› ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                return

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input in ("/quit", "/exit"):
                console.print("[dim]Goodbye.[/dim]")
                return
            if user_input == "/compact":
                ok = await compact_conversation(state, config)
                await storage.flush()
                if ok:
                    console.print("[dim]⤢ Conversation compacted.[/dim]")
                else:
                    console.print("[red]Compaction failed.[/red]")
                continue
            if user_input in ("/clear", "/reset", "/new"):
                await storage.flush()
                storage = _reset_state_for_clear(state, config, settings)
                await storage.flush()
                console.print(f"[dim]Conversation cleared. New session: {storage.session_id}[/dim]")
                continue
            if user_input in ("/help", "/?"):
                console.print(format_help(SKILL_REGISTRY, AGENT_REGISTRY))
                continue
            if user_input == "/cost":
                console.print(format_cost(state.token_usage, config.model))
                continue
            if user_input == "/model":
                console.print(format_model(config.model, config.context_window))
                continue
            if user_input == "/init":
                console.print("[dim]Analyzing codebase to initialize CLAUDE.md...[/dim]")
                user_input = INIT_PROMPT

            # A user-defined /command expands into a prompt (and may restrict tools).
            allowed_tools: list[str] | None = None
            if user_input.startswith("/"):
                dispatch = await dispatch_skill(
                    user_input, SkillContext(cwd=config.cwd, session_id=storage.session_id)
                )
                if dispatch is not None:
                    name = user_input[1:].split(" ", 1)[0]
                    console.print(f"[dim]▶ Running /{name} skill.[/dim]")
                    user_input = dispatch.prompt
                    allowed_tools = dispatch.allowed_tools

            user_msg = {"role": "user", "content": user_input}
            state.messages.append(user_msg)
            storage.append_message(user_msg)
            state.cancel_event.clear()

            try:
                result = await query_loop(
                    state,
                    config,
                    settings=settings,
                    prompter=prompter,
                    callbacks=ui.callbacks(),
                    allowed_tools=allowed_tools,
                )
            except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
                console.print(f"\n[red]Error:[/red] {exc}")
                continue
            finally:
                ui.finish_turn()
                await storage.flush()

            if result.reason is StopReason.COMPLETED:
                stop = await execute_hooks(
                    HookEvent.STOP, session_id=storage.session_id, cwd=config.cwd
                )
                for warning in stop.warnings:
                    console.print(f"[yellow]Stop hook: {warning}[/yellow]")

            console.print()
            if result.reason is StopReason.MAX_TURNS:
                console.print("[yellow]Reached max turns.[/yellow]")
            elif result.reason is StopReason.BLOCKED:
                console.print(f"[yellow]{result.final_text}[/yellow]")
            console.print(format_turn_footer(state.token_usage, config.model))
    finally:
        await storage.flush()
        await close_mcp()
        console.print("[dim]Session saved. Resume with `nano-claude --resume`.[/dim]")


def _init_state(config: AgentConfig, settings: Settings, resume: bool) -> LoopState:
    """Build the loop state and storage, either fresh or resumed from disk."""
    state = LoopState()

    chosen = _pick_session(config.cwd) if resume else None
    if chosen is not None:
        storage = SessionStorage(chosen.path, chosen.session_id)
        records = load_records(chosen.path)
        state.messages = repair_messages(restore_messages(records))
        state.read_file_state = restore_read_file_state(state.messages, config.cwd)
        # Seed the microcompact time gate so it measures the gap across the resume.
        state.last_assistant_at = last_assistant_ts(records)
        state.storage = storage
        console.print(
            f"[dim]Resumed {len(state.messages)} message(s) from {chosen.session_id}.[/dim]"
        )
        return state

    session_id = new_session_id()
    storage = SessionStorage(session_file(config.cwd, session_id), session_id)
    storage.append_metadata(model=config.model, cwd=config.cwd)
    system_msg = {"role": "system", "content": build_system_prompt(config.cwd, settings)}
    state.messages.append(system_msg)
    storage.append_message(system_msg)
    state.storage = storage
    return state


def _reset_state_for_clear(
    state: LoopState, config: AgentConfig, settings: Settings
) -> SessionStorage:
    """Start a fresh session after /clear while keeping the current config."""
    session_id = new_session_id()
    storage = SessionStorage(session_file(config.cwd, session_id), session_id)
    storage.append_metadata(model=config.model, cwd=config.cwd)
    system_msg = {"role": "system", "content": build_system_prompt(config.cwd, settings)}

    state.messages = [system_msg]
    state.turn_count = 0
    state.token_usage = TokenUsage()
    state.last_input_tokens = 0
    state.cancel_event.clear()
    state.consecutive_compact_failures = 0
    state.storage = storage
    state.budget = None
    state.collapse = None
    state.last_assistant_at = None
    state.read_file_state = {}
    storage.append_message(system_msg)
    return storage


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="LiteLLM model string.")
@click.option("--max-turns", default=50, show_default=True, help="Hard cap on loop iterations.")
@click.option(
    "--permission-mode",
    type=click.Choice([m.value for m in PermissionMode]),
    default=PermissionMode.DEFAULT.value,
    show_default=True,
    help="Permission mode: default | acceptEdits | bypassPermissions.",
)
@click.option("--resume", is_flag=True, help="Resume a previous session in this directory.")
@click.option(
    "--context-collapse",
    is_flag=True,
    help="Enable Layer 4 context collapse (experimental): summarize old read/search spans.",
)
@click.option(
    "--tool-preview-format",
    type=click.Choice(["prefix", "head_tail"]),
    default="prefix",
    show_default=True,
    help="How over-budget tool results are previewed: keep the head (prefix) or head+tail.",
)
def cli(
    model: str,
    max_turns: int,
    permission_mode: str,
    resume: bool,
    context_collapse: bool,
    tool_preview_format: str,
) -> None:
    """nano-claude-code: a minimal Claude Code clone."""
    settings = Settings.load()
    load_agents()  # populate AGENT_REGISTRY (built-in + ~/.nano-claude/agents/*.md)
    config = AgentConfig(
        model=model,
        max_turns=max_turns,
        permission_mode=PermissionMode(permission_mode),
        context_collapse=context_collapse,
        tool_result_preview_format=tool_preview_format,
    )
    config.context_window = _resolve_context_window(model, config.context_window)

    state = _init_state(config, settings, resume)

    try:
        asyncio.run(_repl(config, settings, state))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
