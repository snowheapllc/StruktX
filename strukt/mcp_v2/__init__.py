"""
FastMCP v2 integration for StruktX.

This module provides FastMCP v2 integration for StruktX, exposing handlers with
mcp_* prefixed methods as MCP tools. It maintains Strukt's framework-agnostic
philosophy while leveraging FastMCP's modern features.

Key Features:
- Auto-discovery of mcp_* methods on handlers
- Type-safe schema generation from Python type hints
- Middleware bridging from Strukt to FastMCP
- Multiple transport support (stdio, SSE, HTTP)
- Authentication integration

See README.md for comprehensive documentation and examples.
"""

from __future__ import annotations

from .fastmcp_server import StruktFastMCPServer
from .handler_adapters import discover_mcp_methods, build_fastmcp_tool_from_method
from .transports import run_stdio, run_sse, run_http
from .asgi import build_fastapi_app, create_mcp_v2_app
from .auth import APIKeyAuthConfig, APIKeyAuthorizer

__all__ = [
    "StruktFastMCPServer",
    "discover_mcp_methods",
    "build_fastmcp_tool_from_method",
    "run_stdio",
    "run_sse",
    "run_http",
    "build_fastapi_app",
    "create_mcp_v2_app",
    "APIKeyAuthConfig",
    "APIKeyAuthorizer",
]
