"""CLI entry point and interactive REPL.

Phase 1: a bare conversation loop (no tools). Reads user input, runs the agent
loop, streams the assistant's reply, and repeats until the user exits.
"""

from __future__ import annotations

import asyncio
import os

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console

from nano_claude.agent.loop import query_loop
from nano_claude.agent.types import AgentConfig, LoopState, StopReason
from nano_claude.context import build_system_prompt
from nano_claude.permissions.modes import PermissionMode

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


async def _repl(config: AgentConfig) -> None:
    state = LoopState()
    state.messages.append({"role": "system", "content": build_system_prompt()})

    session: PromptSession = PromptSession()

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

        first = True

        def on_text(delta: str) -> None:
            nonlocal first
            if first:
                console.print("[bold green]assistant[/bold green]")
                first = False
            console.print(delta, end="", markup=False, highlight=False)

        try:
            result = await query_loop(state, config, on_text=on_text)
        except Exception as exc:  # noqa: BLE001 - surface any provider error to the user
            console.print(f"\n[red]Error:[/red] {exc}")
            # Drop the unanswered user turn so the next request stays valid.
            state.messages.pop()
            continue

        console.print()  # newline after streamed output
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
    help="Permission mode (tools land in Phase 2).",
)
def cli(model: str, max_turns: int, permission_mode: str) -> None:
    """nano-claude-code: a minimal Claude Code clone."""
    config = AgentConfig(
        model=model,
        max_turns=max_turns,
        permission_mode=PermissionMode(permission_mode),
    )
    config.context_window = _resolve_context_window(model, config.context_window)

    try:
        asyncio.run(_repl(config))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
