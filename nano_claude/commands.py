"""Built-in slash commands for the REPL.

The state-mutating commands (``/clear``, ``/compact``, ``/quit``) are handled
inline in the REPL because they rebind session state. The read-only commands
here are pure ``format_*`` functions returning the text to print, so they can be
unit-tested without a live session.
"""

from __future__ import annotations

from nano_claude.agent.types import TokenUsage
from nano_claude.extensibility.skills.types import SkillDefinition
from nano_claude.subagents.types import AgentDefinition

# (command, description) pairs shown by /help. Skills and agents are listed
# separately because they're discovered at runtime.
BUILTIN_COMMANDS: list[tuple[str, str]] = [
    ("/help", "Show this help."),
    ("/cost", "Show token usage and estimated cost for this session."),
    ("/model", "Show the current model and context window."),
    ("/compact", "Summarize the conversation so far to free up context."),
    ("/clear", "Clear the conversation and start a fresh session."),
    ("/init", "Analyze the codebase and draft a CLAUDE.md."),
    ("/quit", "Exit nano-claude-code."),
]


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
    return f"[bold]Model[/bold] {model}\n  Context window: {context_window:,} tokens"


def format_turn_footer(usage: TokenUsage, model: str) -> str:
    """A compact one-line running total shown after each turn."""
    cost = estimate_cost(usage, model)
    parts = [f"{usage.total:,} tokens"]
    if cost is not None:
        parts.append(f"${cost:.4f}")
    return "[dim]· " + " · ".join(parts) + "[/dim]"
