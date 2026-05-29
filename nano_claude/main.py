"""CLI entry point and interactive REPL.

Phase 2: the loop runs tools. The REPL loads persistent permission settings,
wires an interactive permission prompter, and renders tool activity as it
happens.
"""

from __future__ import annotations

import asyncio
import os

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from nano_claude.agent.loop import LoopCallbacks, query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.context import build_system_prompt
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.prompt import make_cli_prompter
from nano_claude.permissions.settings import Settings
from nano_claude.tools.base import ToolResult

DEFAULT_MODEL = os.environ.get("NANO_CLAUDE_MODEL", "anthropic/claude-sonnet-4-6")

console = Console()


def _resolve_context_window(model: str, fallback: int) -> int:
    """Look up the model's context window via LiteLLM, falling back on failure."""
    try:
        import litellm

        info = litellm.get_model_info(model)
    except Exception:
        return fallback
    window = (info or {}).get("max_input_tokens") or (info or {}).get("max_tokens")
    return int(window) if window else fallback


def _summarize_args(args: dict) -> str:
    for key in ("command", "file_path", "pattern", "path"):
        if key in args and args[key]:
            return str(args[key])
    return ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])


def _make_callbacks() -> LoopCallbacks:
    state = {"streaming": False}

    def on_assistant_start() -> None:
        console.print("[bold green]assistant[/bold green]")
        state["streaming"] = True

    def on_text(delta: str) -> None:
        console.print(delta, end="", markup=False, highlight=False)

    def _end_stream() -> None:
        if state["streaming"]:
            console.print()
            state["streaming"] = False

    def on_tool_start(name: str, args: dict) -> None:
        _end_stream()
        console.print(f"[cyan]⚙ {name}[/cyan]([dim]{_summarize_args(args)}[/dim])")

    def on_tool_end(name: str, result: ToolResult) -> None:
        style = "red" if result.is_error else "dim"
        preview = result.output.strip().splitlines()
        head = preview[0] if preview else ""
        more = f" (+{len(preview) - 1} more lines)" if len(preview) > 1 else ""
        console.print(f"  [{style}]{head}{more}[/{style}]")

    def on_tool_denied(name: str, reason: str) -> None:
        _end_stream()
        console.print(f"[yellow]✗ {name} denied[/yellow] [dim]({reason})[/dim]")

    return LoopCallbacks(
        on_text=on_text,
        on_assistant_start=on_assistant_start,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_denied=on_tool_denied,
    )


async def _repl(config: AgentConfig, settings: Settings) -> None:
    state = LoopState()
    state.messages.append({"role": "system", "content": build_system_prompt(config.cwd)})

    session: PromptSession = PromptSession()
    prompter = make_cli_prompter(session)

    console.print(
        f"[bold cyan]nano-claude-code[/bold cyan] [dim]({config.model}, "
        f"mode={config.permission_mode.value})[/dim]"
    )
    console.print("[dim]Type your message. /quit or Ctrl-D to exit.[/dim]\n")

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

        state.messages.append({"role": "user", "content": user_input})
        state.cancel_event.clear()

        try:
            result = await query_loop(
                state,
                config,
                settings=settings,
                prompter=prompter,
                callbacks=_make_callbacks(),
            )
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
            console.print(f"\n[red]Error:[/red] {exc}")
            # Drop the unanswered user turn so the next request stays valid.
            state.messages.pop()
            continue

        console.print()
        if result.reason is StopReason.MAX_TURNS:
            console.print("[yellow]Reached max turns.[/yellow]")


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
def cli(model: str, max_turns: int, permission_mode: str) -> None:
    """nano-claude-code: a minimal Claude Code clone."""
    settings = Settings.load()
    config = AgentConfig(
        model=model,
        max_turns=max_turns,
        permission_mode=PermissionMode(permission_mode),
    )
    config.context_window = _resolve_context_window(model, config.context_window)

    try:
        asyncio.run(_repl(config, settings))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
