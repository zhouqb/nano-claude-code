"""Built-in slash commands for the REPL.

The state-mutating commands (``/clear``, ``/compact``, ``/quit``) are handled
inline in the REPL because they rebind session state. The read-only commands
here are pure ``format_*`` functions returning the text to print, so they can be
unit-tested without a live session.
"""

from __future__ import annotations

from pathlib import Path

from nano_claude.agent.types import TokenUsage
from nano_claude.extensibility.skills.types import SkillDefinition
from nano_claude.memory.paths import ENTRYPOINT
from nano_claude.memory.scan import scan_memory_files
from nano_claude.subagents.types import AgentDefinition

# (command, description) pairs shown by /help. Skills and agents are listed
# separately because they're discovered at runtime.
BUILTIN_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help."),
    ("/cost", "Show token usage and estimated cost for this session."),
    ("/model", "Show the model, or switch it: /model <litellm-model-string>."),
    ("/compact", "Summarize the conversation so far to free up context."),
    ("/clear", "Clear the conversation and start a fresh session."),
    ("/memory", "List memories; /memory <file> opens it in $EDITOR."),
    ("/remember", "Save a fact to memory (e.g. /remember we deploy on Fridays)."),
    ("/forget", "Delete a memory by topic (e.g. /forget deploy schedule)."),
    ("/init", "Analyze the codebase and draft a CLAUDE.md."),
    ("/quit", "Exit nano-claude-code."),
]


def format_memory(mdir: Path | None) -> str:
    """Render the /memory listing: where memory lives and what's saved."""
    if mdir is None:
        return "[yellow]Memory is disabled for this session.[/yellow]"
    headers = scan_memory_files(mdir)
    lines = [f"[bold]Memory[/bold] [dim]{mdir}[/dim]"]
    if not headers:
        lines.append("  [dim](no memories yet)[/dim]")
    else:
        width = max(len(h.filename) for h in headers)
        for h in headers:
            tag = f"[{h.type}] " if h.type else ""
            desc = h.description or ""
            lines.append(f"  [cyan]{h.filename.ljust(width)}[/cyan]  [dim]{tag}[/dim]{desc}")
    lines.append(
        "[dim]Open one with /memory <file> (creates it if missing), or use /remember.[/dim]"
    )
    return "\n".join(lines)


def memory_target_path(mdir: Path, arg: str) -> Path:
    """The file ``/memory [arg]`` edits: a named topic file, else the index.

    The argument is reduced to a bare filename (any directory parts stripped) and
    given a ``.md`` suffix, so it can never escape the memory directory.
    """
    name = arg.strip()
    if not name:
        return mdir / ENTRYPOINT
    safe = Path(name).name
    if not safe.endswith(".md"):
        safe += ".md"
    return mdir / safe


def open_memory_file(mdir: Path, arg: str, *, editor=None) -> Path:
    """Open (creating if needed) a memory file in the user's editor.

    ``editor`` defaults to ``click.edit`` and is injectable for testing. Returns
    the resolved target path.
    """
    if editor is None:
        from click import edit as editor
    target = memory_target_path(mdir, arg)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")  # so the editor opens a real file
    editor(filename=str(target))
    return target


def remember_directive(text: str) -> str:
    """The instruction /remember hands to the agent to save a fact itself."""
    return (
        "Save the following to your memory as the most appropriate type "
        f"(user / feedback / project / reference), then confirm in one line:\n\n{text}"
    )


def forget_directive(topic: str) -> str:
    """The instruction /forget hands to the agent to remove a memory."""
    return (
        f"Find the memory about '{topic}' and delete it — remove both the topic "
        "file and its MEMORY.md pointer. If nothing matches, say so instead."
    )


def format_help(
    skills: dict[str, SkillDefinition] | None = None,
    agents: dict[str, AgentDefinition] | None = None,
) -> str:
    """Render the help text: built-in commands, then skills, then agents."""
    lines = ["[bold]Commands[/bold]"]
    width = max(len(name) for name, _ in BUILTIN_COMMANDS)
    for name, desc in BUILTIN_COMMANDS:
        lines.append(f"  [cyan]{name.ljust(width)}[/cyan]  {desc}")

    if skills:
        lines.append("")
        lines.append("[bold]Skills[/bold] [dim](user-defined /commands)[/dim]")
        swidth = max(len(n) for n in skills)
        for name, skill in sorted(skills.items()):
            hint = f" [dim]{skill.argument_hint}[/dim]" if skill.argument_hint else ""
            lines.append(f"  [cyan]/{name.ljust(swidth)}[/cyan]  {skill.description}{hint}")

    if agents:
        lines.append("")
        lines.append("[bold]Agents[/bold] [dim](delegate via the Task tool)[/dim]")
        awidth = max(len(n) for n in agents)
        for name, agent in sorted(agents.items()):
            lines.append(f"  [green]{name.ljust(awidth)}[/green]  {agent.description}")

    return "\n".join(lines)


def estimate_cost(usage: TokenUsage, model: str) -> float | None:
    """Best-effort USD estimate from input/output tokens; None if model unknown."""
    try:
        import litellm

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.input_tokens + usage.cache_read_tokens,
            completion_tokens=usage.output_tokens,
        )
    except Exception:
        return None
    return prompt_cost + completion_cost


def format_cost(usage: TokenUsage, model: str) -> str:
    """Render the /cost report."""
    lines = [
        "[bold]Token usage this session[/bold]",
        f"  Input:      {usage.input_tokens:,}",
        f"  Output:     {usage.output_tokens:,}",
        f"  Cache read: {usage.cache_read_tokens:,}",
        f"  Total:      {usage.total:,}",
    ]
    cost = estimate_cost(usage, model)
    if cost is None:
        lines.append("  [dim]Cost: unavailable for this model.[/dim]")
    else:
        lines.append(f"  Estimated cost: [green]${cost:.4f}[/green] [dim](approximate)[/dim]")
    return "\n".join(lines)


def format_model(model: str, context_window: int) -> str:
    """Render the /model summary."""
    return (
        f"[bold]Model[/bold] {model}\n"
        f"  Context window: {context_window:,} tokens\n"
        "[dim]Switch with /model <litellm-model-string> (e.g. deepseek/deepseek-chat).[/dim]"
    )


def model_supports_function_calling(model: str) -> bool | None:
    """Whether LiteLLM reports tool-calling support, or None if the model is unknown.

    The agent loop drives everything through tool calls, so a model that can't
    do function calling is effectively unusable here — surfaced as a warning on
    switch rather than a hard error (LiteLLM's map isn't exhaustive)."""
    try:
        import litellm

        info = litellm.get_model_info(model)
    except Exception:
        return None
    return (info or {}).get("supports_function_calling")


def format_model_switch(model: str, context_window: int, supports_tools: bool | None) -> str:
    """Render the confirmation for ``/model <name>``."""
    lines = [
        f"[green]Switched model →[/green] [bold]{model}[/bold]",
        f"  Context window: {context_window:,} tokens",
    ]
    if supports_tools is False:
        lines.append(
            "  [yellow]⚠ This model does not advertise tool-calling support; "
            "the agent may be unable to run tools.[/yellow]"
        )
    elif supports_tools is None:
        lines.append(
            "  [dim]Tool-calling support unknown to LiteLLM — proceed if you trust it.[/dim]"
        )
    return "\n".join(lines)


def format_turn_footer(usage: TokenUsage, model: str) -> str:
    """A compact one-line running total shown after each turn."""
    cost = estimate_cost(usage, model)
    parts = [f"{usage.total:,} tokens"]
    if cost is not None:
        parts.append(f"${cost:.4f}")
    return "[dim]· " + " · ".join(parts) + "[/dim]"
