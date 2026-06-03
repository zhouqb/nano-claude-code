"""Terminal UI for the REPL: spinner, streaming text, tool display, footer.

``ReplUI`` owns the rich rendering and exposes the loop's display callbacks as
methods. It keeps a single spinner running while the model is working and stops
it the instant output (text or a tool call) begins, so the spinner and streamed
text never fight over the terminal. All spinner operations are defensive — a UI
glitch must never crash the loop.
"""

from __future__ import annotations

from rich.console import Console

from nano_claude.agent.loop import LoopCallbacks
from nano_claude.tools.base import ToolResult

# Tool args worth surfacing in the one-line tool header, in priority order.
_ARG_KEYS = ("command", "file_path", "pattern", "path", "subagent_type")


def _summarize_args(args: dict) -> str:
    for key in _ARG_KEYS:
        if args.get(key):
            return str(args[key])
    return ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])


class ReplUI:
    """Renders the agent loop's events; hand ``callbacks()`` to ``query_loop``."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._status = None  # rich Status while the model is working
        self._streaming = False

    # --- spinner ------------------------------------------------------------

    def _start_spinner(self, text: str = "Thinking…") -> None:
        if self._status is not None:
            return
        try:
            self._status = self.console.status(text, spinner="dots")
            self._status.start()
        except Exception:  # noqa: BLE001 - the spinner is cosmetic; never crash
            self._status = None

    def _stop_spinner(self) -> None:
        if self._status is None:
            return
        try:
            self._status.stop()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._status = None

    def _end_stream(self) -> None:
        if self._streaming:
            self.console.print()
            self._streaming = False

    # --- loop callbacks -----------------------------------------------------

    def on_request_start(self) -> None:
        self._start_spinner()

    def on_assistant_start(self) -> None:
        self._stop_spinner()
        self.console.print("[bold green]assistant[/bold green]")
        self._streaming = True

    def on_text(self, delta: str) -> None:
        self.console.print(delta, end="", markup=False, highlight=False)

    def on_tool_start(self, name: str, args: dict) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print(f"[cyan]⚙ {name}[/cyan]([dim]{_summarize_args(args)}[/dim])")

    def on_tool_end(self, name: str, result: ToolResult) -> None:
        style = "red" if result.is_error else "dim"
        lines = result.output.strip().splitlines()
        head = lines[0] if lines else ""
        more = f" (+{len(lines) - 1} more lines)" if len(lines) > 1 else ""
        self.console.print(f"  [{style}]{head}{more}[/{style}]")

    def on_tool_denied(self, name: str, reason: str) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print(f"[yellow]✗ {name} denied[/yellow] [dim]({reason})[/dim]")

    def on_compact(self) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print("[dim]⤢ Context auto-compacted.[/dim]")

    def on_compact_disabled(self) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print("[red]Auto-compact disabled after repeated failures.[/red]")

    def on_context_warning(self) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print("[yellow]Context nearing limit.[/yellow]")

    def on_snip(self, removed: int) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print(f"[dim]✂ Snipped {removed} stale message(s).[/dim]")

    def on_collapse(self) -> None:
        self._stop_spinner()
        self._end_stream()
        self.console.print("[dim]⊟ Collapsed earlier read/search activity.[/dim]")

    def pause_for_input(self) -> None:
        """Release the terminal before a blocking prompt (e.g. permission ask).

        The spinner is a rich Live display; leaving it running while
        prompt_toolkit draws a prompt makes the two fight over the terminal and
        the prompt looks hung. Stop the spinner and close any open stream first.
        """
        self._stop_spinner()
        self._end_stream()

    # --- lifecycle ----------------------------------------------------------

    def finish_turn(self) -> None:
        """Call after query_loop returns: end any open stream and stop the spinner."""
        self._stop_spinner()
        self._end_stream()

    def callbacks(self) -> LoopCallbacks:
        return LoopCallbacks(
            on_text=self.on_text,
            on_request_start=self.on_request_start,
            on_assistant_start=self.on_assistant_start,
            on_tool_start=self.on_tool_start,
            on_tool_end=self.on_tool_end,
            on_tool_denied=self.on_tool_denied,
            on_compact=self.on_compact,
            on_compact_disabled=self.on_compact_disabled,
            on_context_warning=self.on_context_warning,
            on_snip=self.on_snip,
            on_collapse=self.on_collapse,
        )
