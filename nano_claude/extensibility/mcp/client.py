"""MCP client: connect configured servers and expose their tools.

nano-claude is an MCP *client*. At startup it connects to each configured
server, lists its tools, and wraps each as an :class:`MCPTool`. Sessions must
outlive the connect call, so they are owned by a process-lifetime
``AsyncExitStack`` that is entered inside the event loop and closed in the same
task on shutdown (``close_mcp``) — entering and closing from one task avoids
the anyio cancel-scope error the MCP SDK raises otherwise.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack

from rich.console import Console

from nano_claude.extensibility.mcp.tool import MCPTool
from nano_claude.extensibility.mcp.types import (
    MCPServerConfig,
    StdioServerConfig,
    parse_server_config,
)

_console = Console()

# Process-lifetime stack owning every open MCP session. Closed by close_mcp().
_STACK = AsyncExitStack()


def _expand_headers(headers: dict[str, str]) -> dict[str, str]:
    """Interpolate ``$VAR`` references in header values from the environment."""
    return {k: os.path.expandvars(v) for k, v in headers.items()}


async def connect_server(name: str, cfg: MCPServerConfig) -> list[MCPTool]:
    """Connect one server, initialize the session, and wrap its tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if isinstance(cfg, StdioServerConfig):
        read, write = await _STACK.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=cfg.command,
                    args=cfg.args,
                    env={**os.environ, **cfg.env},
                )
            )
        )
    else:  # HTTPServerConfig — "http" | "sse"
        from mcp.client.streamable_http import streamablehttp_client

        read, write, _ = await _STACK.enter_async_context(
            streamablehttp_client(cfg.url, headers=_expand_headers(cfg.headers))
        )

    session = await _STACK.enter_async_context(ClientSession(read, write))
    await session.initialize()
    listed = await session.list_tools()
    return [MCPTool(name, session, spec) for spec in listed.tools]


async def load_mcp_servers(mcp_servers: dict[str, dict]) -> list[MCPTool]:
    """Connect every configured server; one bad server never aborts startup."""
    tools: list[MCPTool] = []
    for name, raw in mcp_servers.items():
        try:
            cfg = parse_server_config(raw)
            tools.extend(await connect_server(name, cfg))
        except Exception as exc:  # noqa: BLE001 - isolate per-server failures
            _console.print(f"[yellow]MCP server '{name}' failed: {exc}[/yellow]")
    return tools


async def close_mcp() -> None:
    """Close every open MCP session/subprocess. Call once on shutdown."""
    await _STACK.aclose()
