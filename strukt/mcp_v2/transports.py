"""
Transport utilities for FastMCP v2 integration.

This module provides utilities for running FastMCP servers with different
transport protocols (stdio, SSE, etc.).
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

from .fastmcp_server import StruktFastMCPServer
from ..logging import get_logger

_log = get_logger("mcp_v2.transports")


def run_stdio(server: StruktFastMCPServer) -> None:
    """Run FastMCP server over stdio transport.

    Args:
        server: The StruktFastMCPServer instance to run
    """
    _log.info("Starting FastMCP server over stdio transport")

    try:
        # FastMCP stdio transport
        server.mcp.run()
    except KeyboardInterrupt:
        _log.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        _log.error(f"Server error: {e}")
        raise


def run_sse(
    server: StruktFastMCPServer, host: str = "localhost", port: int = 8000
) -> None:
    """Run FastMCP server over SSE transport.

    Args:
        server: The StruktFastMCPServer instance to run
        host: Host to bind to
        port: Port to bind to
    """
    _log.info(f"Starting FastMCP server over SSE transport at {host}:{port}")

    try:
        # FastMCP SSE transport - exact API may vary by version
        if hasattr(server.mcp, "run"):
            # Try different parameter patterns based on FastMCP version
            try:
                server.mcp.run(transport="sse", host=host, port=port)
            except TypeError:
                # Fallback for different API
                try:
                    server.mcp.run(host=host, port=port)
                except TypeError:
                    # Another fallback - some versions might use different method names
                    if hasattr(server.mcp, "serve"):
                        server.mcp.serve(host=host, port=port)
                    else:
                        raise RuntimeError(
                            "FastMCP instance does not support SSE transport"
                        )
        else:
            raise RuntimeError("FastMCP instance does not support run() method")
    except KeyboardInterrupt:
        _log.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        _log.error(f"Server error: {e}")
        raise


def run_http(
    server: StruktFastMCPServer, host: str = "localhost", port: int = 8000
) -> None:
    """Run FastMCP server over HTTP transport.

    Args:
        server: The StruktFastMCPServer instance to run
        host: Host to bind to
        port: Port to bind to
    """
    _log.info(f"Starting FastMCP server over HTTP transport at {host}:{port}")

    try:
        # FastMCP HTTP transport
        if hasattr(server.mcp, "run"):
            try:
                server.mcp.run(transport="http", host=host, port=port)
            except TypeError:
                # Fallback for different API
                try:
                    server.mcp.run(host=host, port=port)
                except TypeError:
                    if hasattr(server.mcp, "serve"):
                        server.mcp.serve(host=host, port=port)
                    else:
                        raise RuntimeError(
                            "FastMCP instance does not support HTTP transport"
                        )
        else:
            raise RuntimeError("FastMCP instance does not support run() method")
    except KeyboardInterrupt:
        _log.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        _log.error(f"Server error: {e}")
        raise


def run_with_signal_handling(
    server: StruktFastMCPServer, transport: str = "stdio", **kwargs
) -> None:
    """Run server with proper signal handling for graceful shutdown.

    Args:
        server: The StruktFastMCPServer instance to run
        transport: Transport type ("stdio", "sse", "http")
        **kwargs: Additional arguments for the transport
    """
    _log.info(f"Starting FastMCP server with {transport} transport and signal handling")

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        _log.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if transport == "stdio":
            run_stdio(server)
        elif transport == "sse":
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 8000)
            run_sse(server, host=host, port=port)
        elif transport == "http":
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 8000)
            run_http(server, host=host, port=port)
        else:
            raise ValueError(f"Unsupported transport: {transport}")
    except Exception as e:
        _log.error(f"Server error: {e}")
        raise


def run_async(server: StruktFastMCPServer, transport: str = "stdio", **kwargs) -> None:
    """Run server asynchronously.

    Args:
        server: The StruktFastMCPServer instance to run
        transport: Transport type ("stdio", "sse", "http")
        **kwargs: Additional arguments for the transport
    """
    _log.info(f"Starting FastMCP server asynchronously with {transport} transport")

    async def async_run():
        try:
            if transport == "stdio":
                # For stdio, we still need to run synchronously
                run_stdio(server)
            elif transport == "sse":
                host = kwargs.get("host", "localhost")
                port = kwargs.get("port", 8000)
                run_sse(server, host=host, port=port)
            elif transport == "http":
                host = kwargs.get("host", "localhost")
                port = kwargs.get("port", 8000)
                run_http(server, host=host, port=port)
            else:
                raise ValueError(f"Unsupported transport: {transport}")
        except Exception as e:
            _log.error(f"Async server error: {e}")
            raise

    try:
        asyncio.run(async_run())
    except KeyboardInterrupt:
        _log.info("Async server stopped by user")
    except Exception as e:
        _log.error(f"Async server error: {e}")
        raise


def get_available_transports() -> list[str]:
    """Get list of available transport types.

    Returns:
        List of supported transport names
    """
    return ["stdio", "sse", "http"]


def validate_transport(transport: str) -> bool:
    """Validate that a transport type is supported.

    Args:
        transport: Transport type to validate

    Returns:
        True if transport is supported, False otherwise
    """
    return transport in get_available_transports()


def get_transport_defaults(transport: str) -> dict[str, Any]:
    """Get default configuration for a transport type.

    Args:
        transport: Transport type

    Returns:
        Dictionary of default configuration values
    """
    defaults = {
        "stdio": {},
        "sse": {"host": "localhost", "port": 8000},
        "http": {"host": "localhost", "port": 8000},
    }

    return defaults.get(transport, {})
