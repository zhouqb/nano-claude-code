"""Persistent permission settings (~/.nano-claude/settings.json).

Holds the permission mode plus always-allow / always-deny rule lists. "Always"
choices made at the permission prompt are persisted here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nano_claude.extensibility.hooks.types import HookDefinition
from nano_claude.permissions.modes import PermissionMode
from nano_claude.permissions.rules import PermissionRule

DEFAULT_SETTINGS_PATH = Path.home() / ".nano-claude" / "settings.json"

# Tools that are safe enough to allow by default on a fresh install.
DEFAULT_ALLOW_PATTERNS = ["Read", "GlobTool", "Grep"]

# settings.json keys this class owns; everything else (e.g. "hooks", and
# "mcpServers" once later phases add it) is round-tripped verbatim so persisting
# a permission decision never drops unrelated sections of the file.
_MANAGED_KEYS = frozenset({"permissionMode", "alwaysAllowRules", "alwaysDenyRules"})


def _parse_hooks(raw: object) -> list[HookDefinition]:
    """Parse the ``"hooks"`` list, skipping malformed entries."""
    if not isinstance(raw, list):
        return []
    hooks: list[HookDefinition] = []
    for entry in raw:
        try:
            hooks.append(HookDefinition.model_validate(entry))
        except Exception:  # noqa: BLE001 - one bad hook must not break startup
            continue
    return hooks


@dataclass
class Settings:
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    allow_rules: list[PermissionRule] = field(default_factory=list)
    deny_rules: list[PermissionRule] = field(default_factory=list)
    hooks: list[HookDefinition] = field(default_factory=list)
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    path: Path | None = None
    # Unrecognized settings.json keys, preserved across save() so a persisted
    # permission decision can't silently wipe hooks/mcpServers/etc.
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or DEFAULT_SETTINGS_PATH
        if not path.is_file():
            return cls(
                allow_rules=[PermissionRule(p, "allow") for p in DEFAULT_ALLOW_PATTERNS],
                path=path,
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(path=path)

        mode_raw = data.get("permissionMode", PermissionMode.DEFAULT.value)
        try:
            mode = PermissionMode(mode_raw)
        except ValueError:
            mode = PermissionMode.DEFAULT
        return cls(
            permission_mode=mode,
            allow_rules=[PermissionRule(p, "allow") for p in data.get("alwaysAllowRules", [])],
            deny_rules=[PermissionRule(p, "deny") for p in data.get("alwaysDenyRules", [])],
            hooks=_parse_hooks(data.get("hooks", [])),
            mcp_servers=data.get("mcpServers", {})
            if isinstance(data.get("mcpServers"), dict)
            else {},
            extra={k: v for k, v in data.items() if k not in _MANAGED_KEYS},
            path=path,
        )

    def to_json(self) -> str:
        # Start from the managed keys, then re-attach everything else verbatim.
        out = {
            "permissionMode": self.permission_mode.value,
            "alwaysAllowRules": [r.pattern for r in self.allow_rules],
            "alwaysDenyRules": [r.pattern for r in self.deny_rules],
        }
        out.update(self.extra)
        return json.dumps(out, indent=2)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.to_json() + "\n", encoding="utf-8")

    def add_allow_rule(self, pattern: str) -> None:
        if not any(r.pattern == pattern for r in self.allow_rules):
            self.allow_rules.append(PermissionRule(pattern, "allow"))

    def add_deny_rule(self, pattern: str) -> None:
        if not any(r.pattern == pattern for r in self.deny_rules):
            self.deny_rules.append(PermissionRule(pattern, "deny"))
