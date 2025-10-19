"""
ASGI integration for FastMCP v2.

This module provides FastAPI/ASGI integration for FastMCP v2 servers,
replacing the old MCP v1 ASGI implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import json
import asyncio

from pydantic import BaseModel

from ..config import StruktConfig, ensure_config_types
from ..ai import Strukt
from .fastmcp_server import StruktFastMCPServer
from ..logging import get_logger

_log = get_logger("mcp_v2.asgi")


class MCPRequest(BaseModel):
    """MCP JSON-RPC request model."""

    jsonrpc: str = "2.0"
    id: Optional[Any] = None  # Can be string, int, or null
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC response model."""

    jsonrpc: str = "2.0"
    id: Optional[Any] = None  # Can be string, int, or null
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


def build_fastapi_app(
    strukt_app: Strukt,
    cfg: StruktConfig,
    *,
    app: Any | None = None,
    prefix: str = "/mcp",
):
    """Create or extend a FastAPI app exposing FastMCP v2 endpoints.

    This replaces the old MCP v1 ASGI implementation with FastMCP v2 integration.

    Args:
        strukt_app: Strukt application instance
        cfg: Strukt configuration
        app: Optional existing FastAPI app to extend
        prefix: URL prefix for MCP endpoints

    Returns:
        FastAPI app with MCP v2 endpoints
    """
    try:
        from fastapi import (
            FastAPI,
            Header,
            HTTPException,
            APIRouter,
        )
        from fastapi.responses import JSONResponse
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is required to build MCP HTTP endpoints. Install fastapi."
        ) from e

    cfg = ensure_config_types(cfg)

    if app is None:
        app = FastAPI(title="StruktX FastMCP v2 Server")
    router = APIRouter()
    mcp_server: Optional[StruktFastMCPServer] = None

    @app.on_event("startup")
    async def _init_mcp() -> None:
        nonlocal mcp_server
        try:
            handlers = getattr(strukt_app._engine, "_handlers", {})  # type: ignore[attr-defined]
            memory = strukt_app.get_memory()

            # Create FastMCP v2 server
            mcp_server = StruktFastMCPServer(
                handlers=handlers,
                include_handlers=cfg.mcp_v2.include_handlers,
                server_name=cfg.mcp_v2.server_name or "struktx-mcp",
                memory=memory,
                config=cfg.mcp_v2,
                middleware=getattr(strukt_app._engine, "_middleware", []),  # type: ignore[attr-defined]
            )

            _log.info(
                f"Initialized FastMCP v2 server with {len(mcp_server.registered_tools)} tools"
            )

        except Exception as e:
            _log.error(f"Failed to initialize FastMCP v2 server: {e}")
            raise

    def _check_auth(x_api_key: Optional[str]) -> None:
        """Check API key authentication."""
        if not mcp_server:
            raise HTTPException(status_code=503, detail="MCP server not ready")

        if not cfg.mcp_v2.auth_api_key.enabled:
            return  # Auth disabled

        # Use the auth authorizer from the server
        if hasattr(mcp_server, "_auth_authorizer") and mcp_server._auth_authorizer:
            header_name = cfg.mcp_v2.auth_api_key.header_name
            headers = {header_name: x_api_key or ""}
            if not mcp_server._auth_authorizer.is_authorized(headers):
                raise HTTPException(status_code=401, detail="Unauthorized")
        else:
            # Fallback: simple API key check
            import os

            expected_key = os.environ.get(cfg.mcp_v2.auth_api_key.env_var)
            if expected_key and x_api_key != expected_key:
                raise HTTPException(status_code=401, detail="Unauthorized")

    @router.get(f"{prefix}")
    async def list_mcp_tools(
        x_api_key: Optional[str] = Header(None, alias="x-roomi-mcp-api-key"),
    ) -> JSONResponse:
        """List available MCP tools and server information."""
        # Check authentication
        _check_auth(x_api_key)

        if not mcp_server:
            return JSONResponse(
                status_code=503, content={"error": "MCP server not ready"}
            )

        try:
            # Get tools from the server
            tools = mcp_server.list_tools()

            return JSONResponse(
                content={
                    "server_name": mcp_server.server_name,
                    "tools_count": len(tools),
                    "tools": tools,
                    "endpoints": {
                        "jsonrpc": f"{prefix}",
                        "tools": f"{prefix}/tools",
                        "health": f"{prefix}/health",
                    },
                }
            )

        except Exception as e:
            _log.error(f"Failed to list tools: {e}")
            return JSONResponse(
                status_code=500, content={"error": f"Failed to list tools: {str(e)}"}
            )

    @router.post(f"{prefix}")
    async def handle_mcp_request(
        mcp_request: MCPRequest,
        x_api_key: Optional[str] = Header(None, alias="x-roomi-mcp-api-key"),
    ) -> JSONResponse:
        """Handle MCP JSON-RPC requests using FastMCP v2."""
        if not mcp_server:
            return JSONResponse(
                status_code=503,
                content={
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": "MCP server not initialized"},
                },
            )

        try:
            # Check authentication
            _check_auth(x_api_key)

            # Extract request data from Pydantic model
            request_id = mcp_request.id
            method = mcp_request.method
            params = mcp_request.params or {}

            _log.debug(f"MCP request: {method} with params: {params}")

            # Handle different MCP methods
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "prompts": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                            "sampling": {},
                        },
                        "serverInfo": {
                            "name": mcp_server.server_name,
                            "version": "2.0.0",
                        },
                    },
                }
            elif method == "tools/list":
                tools = mcp_server.list_tools()
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools},
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {"code": -32602, "message": "Missing tool name"},
                        },
                    )

                try:
                    # Find the tool in registered tools
                    if tool_name not in mcp_server.registered_tools:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32601,
                                    "message": f"Tool '{tool_name}' not found",
                                },
                            },
                        )

                    # Get the handler for this tool
                    handler_key = mcp_server.registered_tools[tool_name]
                    handler = mcp_server.get_tool_handler(tool_name)

                    if not handler:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32601,
                                    "message": f"Handler for tool '{tool_name}' not found",
                                },
                            },
                        )

                    # Find the mcp_* method on the handler
                    method_name = tool_name.replace(f"{handler_key}_", "mcp_")
                    tool_method = getattr(handler, method_name, None)

                    if not tool_method:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32601,
                                    "message": f"Method '{method_name}' not found on handler '{handler_key}'",
                                },
                            },
                        )

                    # Call the tool method
                    result = (
                        await tool_method(**arguments)
                        if asyncio.iscoroutinefunction(tool_method)
                        else tool_method(**arguments)
                    )

                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2)
                                    if isinstance(result, (dict, list))
                                    else str(result),
                                }
                            ]
                        },
                    }

                except Exception as e:
                    _log.error(f"Tool execution failed: {e}")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Tool execution failed: {str(e)}",
                        },
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not found",
                    },
                }

            return JSONResponse(content=response)

        except HTTPException:
            raise
        except Exception as e:
            _log.error(f"MCP request handling failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "jsonrpc": "2.0",
                    "id": mcp_request.id,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                },
            )

    @router.get(f"{prefix}/tools")
    async def list_tools(
        x_api_key: Optional[str] = Header(None, alias="x-roomi-mcp-api-key"),
    ) -> JSONResponse:
        """List available MCP tools."""
        if not mcp_server:
            return JSONResponse(
                status_code=503, content={"error": "MCP server not initialized"}
            )

        try:
            # Check authentication
            _check_auth(x_api_key)
            tools = mcp_server.list_tools()
            return JSONResponse(content={"tools": tools})
        except HTTPException:
            raise
        except Exception as e:
            _log.error(f"Failed to list tools: {e}")
            return JSONResponse(
                status_code=500, content={"error": f"Failed to list tools: {str(e)}"}
            )

    @router.get(f"{prefix}/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint."""
        return JSONResponse(
            content={
                "status": "healthy",
                "server": mcp_server.server_name if mcp_server else "not_initialized",
                "tools_count": len(mcp_server.registered_tools) if mcp_server else 0,
            }
        )

    # Add routes to the app
    app.include_router(router)

    return app


def create_mcp_v2_app(
    handlers: Dict[str, Any],
    config: StruktConfig,
    *,
    server_name: str | None = None,
    include_handlers: list[str] | None = None,
) -> Any:
    """Create a standalone FastAPI app with FastMCP v2 integration.

    Args:
        handlers: Dictionary of handler instances
        config: Strukt configuration
        server_name: Optional server name override
        include_handlers: Optional list of handlers to include

    Returns:
        FastAPI app with MCP v2 endpoints
    """
    try:
        from fastapi import FastAPI
    except Exception as e:  # pragma: no cover
        raise RuntimeError("FastAPI is required. Install fastapi.") from e

    app = FastAPI(title="StruktX FastMCP v2 Server")
    mcp_server: Optional[StruktFastMCPServer] = None

    @app.on_event("startup")
    async def _init_mcp() -> None:
        nonlocal mcp_server
        try:
            # Create FastMCP v2 server
            mcp_server = StruktFastMCPServer(
                handlers=handlers,
                include_handlers=include_handlers or config.mcp_v2.include_handlers,
                server_name=server_name or config.mcp_v2.server_name or "struktx-mcp",
                config=config.mcp_v2,
            )

            _log.info(
                f"Initialized standalone FastMCP v2 server with {len(mcp_server.registered_tools)} tools"
            )

        except Exception as e:
            _log.error(f"Failed to initialize FastMCP v2 server: {e}")
            raise

    # Add the same routes as build_fastapi_app
    from .asgi import build_fastapi_app

    # Create a mock Strukt app for compatibility
    class MockStrukt:
        def __init__(self, handlers):
            self._engine = type(
                "Engine", (), {"_handlers": handlers, "_middleware": []}
            )()

        def get_memory(self):
            return None

    mock_strukt = MockStrukt(handlers)
    return build_fastapi_app(mock_strukt, config, app=app)
