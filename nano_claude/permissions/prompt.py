"""Interactive permission prompt built on prompt_toolkit.

Produces a :data:`Prompter` the manager can call when a decision is ``ask``.
Kept separate from the manager so the manager stays testable with fake
prompters.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from prompt_toolkit import PromptSession

from nano_claude.permissions.manager import Prompter, PromptOutcome
from nano_claude.tools.base import Tool

_CHOICES = {
    "y": PromptOutcome.ALLOW_ONCE,
    "a": PromptOutcome.ALLOW_ALWAYS,
    "n": PromptOutcome.DENY_ONCE,
    "d": PromptOutcome.DENY_ALWAYS,
}

_MENU = "[y]es once  [a]lways  [n]o  [d]eny always"


def _summarize(args: dict) -> str:
    try:
        text = json.dumps(args)
    except (TypeError, ValueError):
        text = str(args)
    return text if len(text) <= 200 else text[:200] + "…"


def make_cli_prompter(
    session: PromptSession | None = None,
    *,
    on_prompt: Callable[[], None] | None = None,
) -> Prompter:
    """Build a Prompter that asks the user via the terminal.

    ``on_prompt`` is invoked once before the prompt is drawn, so the REPL can
    release the terminal (stop the spinner / end any open stream) before
    prompt_toolkit takes over — otherwise the rich Live spinner and the prompt
    fight over the terminal and the prompt appears to hang.
    """
    session = session or PromptSession()

    async def prompt(tool: Tool, args: dict, prompt_text: str) -> PromptOutcome:
        if on_prompt is not None:
            on_prompt()
        header = prompt_text or f"Allow {tool.name}?"
        print(f"\nPermission required: {header}")
        print(f"  tool: {tool.name}")
        print(f"  args: {_summarize(args)}")
        while True:
            answer = (await session.prompt_async(f"{_MENU} › ")).strip().lower()
            if answer in _CHOICES:
                return _CHOICES[answer]
            print("Please answer y, a, n, or d.")

    return prompt
