"""
MCP integration scaffolding for StruktX.

This module provides provider-agnostic primitives to expose Strukt handlers as
MCP tools. It intentionally avoids coupling to any specific MCP runtime. A
separate runtime adapter (e.g., fast-agent) can import these utilities to host
the server via stdio or HTTP.
"""

from __future__ import annotations
from .adapters import ToolSpec, build_tools_from_handlers
from .permissions import ConsentPolicy, ConsentDecision, ConsentStore
from .auth import APIKeyAuthConfig, APIKeyAuthorizer
from .server import MCPServerApp
from .runtime_fast_agent import FastAgentAdapter
from .asgi import build_fastapi_app

__all__ = [
    "ToolSpec",
    "build_tools_from_handlers",
    "ConsentPolicy",
    "ConsentDecision",
    "ConsentStore",
    "APIKeyAuthConfig",
    "APIKeyAuthorizer",
    "MCPServerApp",
    "FastAgentAdapter",
    "build_fastapi_app",
]
