"""Lifecycle hooks: shell commands fired at points in the agent loop."""

from nano_claude.extensibility.hooks.executor import (
    HookOutcome,
    clear_hooks,
    execute_hooks,
    get_hooks,
    register_hooks,
)
from nano_claude.extensibility.hooks.types import HookDefinition, HookEvent

__all__ = [
    "HookDefinition",
    "HookEvent",
    "HookOutcome",
    "clear_hooks",
    "execute_hooks",
    "get_hooks",
    "register_hooks",
]
