"""MCP server configuration.

Config lives under ``"mcpServers"`` in ``~/.nano-claude/settings.json``, keyed
by server name — the same shape as Claude Code's ``claude mcp add``. A stdio
server is identified by a ``command``; an HTTP/SSE server by a ``transport`` +
``url``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StdioServerConfig(BaseModel):
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class HTTPServerConfig(BaseModel):
    transport: Literal["http", "sse"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)


MCPServerConfig = StdioServerConfig | HTTPServerConfig


def parse_server_config(raw: dict) -> MCPServerConfig:
    """Pick the config shape from a raw settings entry.

    An entry with a ``command`` is stdio; otherwise it must declare a
    ``transport`` + ``url`` for HTTP/SSE.
    """
    if "command" in raw:
        return StdioServerConfig.model_validate(raw)
    return HTTPServerConfig.model_validate(raw)
