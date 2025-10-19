"""
FastMCP v2 server implementation for StruktX.

This module provides the main StruktFastMCPServer class that integrates
Strukt handlers with FastMCP v2, exposing mcp_* methods as MCP tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP is required for mcp_v2 integration. Install with: pip install fastmcp>=2.12.5"
    )

from ..interfaces import Handler, MemoryEngine
from ..config import MCPConfig
from ..logging import get_logger
from .handler_adapters import register_handler_tools
from .middleware_bridge import create_fastmcp_middleware
from .auth_adapter import create_auth_middleware

_log = get_logger("mcp_v2.server")


class StruktFastMCPServer:
    """FastMCP v2 server for StruktX handlers.

    This server automatically discovers and exposes handlers with mcp_* prefixed
    methods as MCP tools, while maintaining Strukt's framework-agnostic design.
    """

    def __init__(
        self,
        *,
        handlers: Dict[str, Handler],
        include_handlers: List[str] | None = None,
        server_name: str = "struktx-mcp",
        memory: MemoryEngine | None = None,
        config: MCPConfig | None = None,
        middleware: List[Any] | None = None,
    ) -> None:
        """Initialize the StruktFastMCPServer.

        Args:
            handlers: Dictionary of handler instances keyed by handler name
            include_handlers: Optional list of handler keys to include (if None, include all)
            server_name: Name for the MCP server
            memory: Optional memory engine for context
            config: Optional MCP configuration
            middleware: Optional list of Strukt middleware to bridge
        """
        self.handlers = handlers
        self.include_handlers = include_handlers
        self.server_name = server_name
        self.memory = memory
        self.config = config or MCPConfig()
        self.middleware = middleware or []

        # Initialize FastMCP instance
        self.mcp = FastMCP(server_name)

        # Track registered tools
        self.registered_tools: Dict[str, str] = {}  # tool_name -> handler_key
        self.tool_descriptions: Dict[str, str] = {}  # tool_name -> description

        # Setup the server
        self._setup_server()

        _log.info(
            f"Initialized StruktFastMCPServer '{server_name}' with {len(self.registered_tools)} tools"
        )

    def _setup_server(self) -> None:
        """Setup the FastMCP server with tools and middleware."""
        # Register tools from all handlers
        self._register_handler_tools()

        # Setup authentication middleware if configured
        self._setup_auth_middleware()

        # Bridge Strukt middleware to FastMCP middleware
        self._setup_middleware_bridge()

    def _register_handler_tools(self) -> None:
        """Register tools from all handlers with mcp_* methods."""
        total_tools = 0

        for handler_key, handler in self.handlers.items():
            try:
                registered_tools = register_handler_tools(
                    fastmcp_instance=self.mcp,
                    handler=handler,
                    handler_key=handler_key,
                    include_handlers=self.include_handlers,
                )

                # Track which handler each tool came from and store descriptions
                for tool_name, description in registered_tools:
                    self.registered_tools[tool_name] = handler_key
                    self.tool_descriptions[tool_name] = description

                total_tools += len(registered_tools)

                if registered_tools:
                    _log.info(
                        f"Registered {len(registered_tools)} tools from handler '{handler_key}'"
                    )
                else:
                    _log.debug(f"No mcp_* methods found on handler '{handler_key}'")

            except Exception as e:
                _log.error(
                    f"Failed to register tools from handler '{handler_key}': {e}"
                )

        _log.info(f"Total registered tools: {total_tools}")

    def _setup_auth_middleware(self) -> None:
        """Setup authentication middleware if API key auth is enabled."""
        if not self.config.auth_api_key.enabled:
            _log.debug("API key authentication disabled")
            self._auth_authorizer = None
            return

        try:
            # Create and store the auth authorizer for direct access
            from .auth import APIKeyAuthorizer

            self._auth_authorizer = APIKeyAuthorizer(self.config.auth_api_key)

            auth_middleware = create_auth_middleware(self.config.auth_api_key)
            # Apply auth middleware to FastMCP
            # Note: FastMCP middleware application may vary by version
            if hasattr(self.mcp, "add_middleware"):
                self.mcp.add_middleware(auth_middleware)
            elif hasattr(self.mcp, "middleware"):
                self.mcp.middleware(auth_middleware)
            else:
                _log.warning(
                    "Could not apply auth middleware - FastMCP version may not support middleware"
                )

            _log.info("Applied API key authentication middleware")

        except Exception as e:
            _log.error(f"Failed to setup auth middleware: {e}")
            self._auth_authorizer = None

    def _setup_middleware_bridge(self) -> None:
        """Bridge Strukt middleware to FastMCP middleware."""
        if not self.middleware:
            _log.debug("No Strukt middleware to bridge")
            return

        try:
            fastmcp_middleware = create_fastmcp_middleware(self.middleware)

            # Apply bridged middleware to FastMCP
            if hasattr(self.mcp, "add_middleware"):
                self.mcp.add_middleware(fastmcp_middleware)
            elif hasattr(self.mcp, "middleware"):
                self.mcp.middleware(fastmcp_middleware)
            else:
                _log.warning(
                    "Could not apply bridged middleware - FastMCP version may not support middleware"
                )

            _log.info(f"Bridged {len(self.middleware)} Strukt middleware to FastMCP")

        except Exception as e:
            _log.error(f"Failed to bridge Strukt middleware: {e}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their specifications."""
        tools = []

        for tool_name, handler_key in self.registered_tools.items():
            # Get tool info from FastMCP
            try:
                # Use the stored description from the tool registration
                description = self.tool_descriptions.get(
                    tool_name, f"Tool from handler '{handler_key}'"
                )

                tool_info = {
                    "name": tool_name,
                    "handler": handler_key,
                    "description": description,
                }
                tools.append(tool_info)
            except Exception as e:
                _log.warning(f"Could not get info for tool '{tool_name}': {e}")

        return tools

    def get_tool_handler(self, tool_name: str) -> Optional[Handler]:
        """Get the handler that provides a specific tool."""
        handler_key = self.registered_tools.get(tool_name)
        if handler_key:
            return self.handlers.get(handler_key)
        return None

    def run_stdio(self) -> None:
        """Run the server over stdio transport."""
        _log.info("Starting FastMCP server over stdio transport")
        try:
            self.mcp.run()
        except KeyboardInterrupt:
            _log.info("Server stopped by user")
        except Exception as e:
            _log.error(f"Server error: {e}")
            raise

    def run_sse(self, host: str = "localhost", port: int = 8000) -> None:
        """Run the server over SSE transport."""
        _log.info(f"Starting FastMCP server over SSE transport at {host}:{port}")
        try:
            # FastMCP SSE transport - exact API may vary by version
            if hasattr(self.mcp, "run"):
                # Try different parameter patterns based on FastMCP version
                try:
                    self.mcp.run(transport="sse", host=host, port=port)
                except TypeError:
                    # Fallback for different API
                    self.mcp.run(host=host, port=port)
            else:
                raise RuntimeError("FastMCP instance does not support run() method")
        except KeyboardInterrupt:
            _log.info("Server stopped by user")
        except Exception as e:
            _log.error(f"Server error: {e}")
            raise

    def get_fastmcp_instance(self) -> FastMCP:
        """Get the underlying FastMCP instance for advanced usage."""
        return self.mcp
