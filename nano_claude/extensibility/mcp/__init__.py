"""MCP (Model Context Protocol): connect external servers as tools."""

from nano_claude.extensibility.mcp.client import (
    close_mcp,
    connect_server,
    load_mcp_servers,
)
from nano_claude.extensibility.mcp.tool import MCPTool
from nano_claude.extensibility.mcp.types import (
    HTTPServerConfig,
    MCPServerConfig,
    StdioServerConfig,
    parse_server_config,
)

__all__ = [
    "HTTPServerConfig",
    "MCPServerConfig",
    "MCPTool",
    "StdioServerConfig",
    "close_mcp",
    "connect_server",
    "load_mcp_servers",
    "parse_server_config",
]
