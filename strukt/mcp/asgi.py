from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import unquote
import json

from ..config import StruktConfig, ensure_config_types
from ..ai import Strukt
from .server import MCPServerApp
from .adapters import build_tools_from_handlers
from .auth import APIKeyAuthConfig


def build_fastapi_app(
    strukt_app: Strukt,
    cfg: StruktConfig,
    *,
    app: Any | None = None,
    prefix: str = "/mcp",
):
    """Create or extend a FastAPI app exposing MCP JSON-RPC protocol endpoints.

    Following MCP specification:
    - POST {prefix} -> handles both tools/list and tools/call requests via JSON-RPC
    - Supports StreamableHTTP for MCP inspector compatibility
    """
    try:
        from fastapi import FastAPI, Header, HTTPException, APIRouter, Request, Response
        from fastapi.responses import StreamingResponse
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is required to build MCP HTTP endpoints. Install fastapi."
        ) from e

    cfg = ensure_config_types(cfg)

    if app is None:
        app = FastAPI()
    router = APIRouter()
    mcp: Optional[MCPServerApp] = None

    @app.on_event("startup")
    async def _init_mcp() -> None:
        nonlocal mcp
        handlers = getattr(strukt_app._engine, "_handlers", {})  # type: ignore[attr-defined]
        memory = strukt_app.get_memory()
        mcp = MCPServerApp(
            server_name=cfg.mcp.server_name or "struktmcp",
            handlers=handlers,
            include_handlers=cfg.mcp.include_handlers,
            memory=memory,
            api_key_auth=APIKeyAuthConfig(
                header_name=cfg.mcp.auth_api_key.header_name,
                env_var=cfg.mcp.auth_api_key.env_var,
            ),
        )
        # Build tools from config
        mcp._tools = build_tools_from_handlers(
            handlers=handlers,
            include=cfg.mcp.include_handlers,
            mcp_config=cfg.mcp,
        )

    def _check_auth(x_api_key: Optional[str]) -> None:
        """Check API key authentication."""
        if not mcp:
            raise HTTPException(status_code=503, detail="MCP app not ready")
        header_name = cfg.mcp.auth_api_key.header_name
        if not mcp.check_api_key({header_name: x_api_key or ""}):
            raise HTTPException(status_code=401, detail="Unauthorized")

    async def _handle_mcp_request(
        request_body: Dict[str, Any], request_id: Any
    ) -> Dict[str, Any] | None:
        """Handle MCP JSON-RPC request using the official MCP server."""
        if not mcp:
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {"code": -32603, "message": "MCP server not initialized"},
            }

        try:
            # For now, fall back to the existing methods until we can properly integrate
            # the official MCP server's transport layer
            method = request_body.get("method")
            params = request_body.get("params", {})

            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "prompts": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                            "sampling": {},
                        },
                        "serverInfo": {
                            "name": mcp.server_name,
                            "version": "1.0.0",
                        },
                    },
                }
            elif method == "notifications/initialized":
                # Notification, no response needed
                return None
            elif method == "tools/list":
                tools = mcp.list_tools()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": {"tools": tools},
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                if not tool_name:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id if request_id is not None else 0,
                        "error": {"code": -32602, "message": "Missing tool name"},
                    }

                result = await mcp.call_tool(tool_name=tool_name, args=arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": result,
                }
            elif method in ["prompts/list", "resources/list"]:
                # Return empty lists for unsupported features
                resource_type = "prompts" if "prompts" in method else "resources"
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": {resource_type: []},
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    @router.get("")
    async def mcp_get_endpoint(
        request: Request,
        x_api_key: Optional[str] = Header(default=None),
    ) -> Response:
        """Handle MCP GET requests for StreamableHTTP compatibility."""
        _check_auth(x_api_key)

        # Check if this is a StreamableHTTP connection
        accept = request.headers.get("accept", "")
        user_agent = request.headers.get("user-agent", "")

        # Handle StreamableHTTP connection
        if "text/event-stream" in accept or "streamable" in user_agent.lower():

            async def streamable_response():
                """Generate StreamableHTTP response stream."""
                yield 'data: {"type":"connection_established"}\n\n'
                try:
                    query_params = dict(request.query_params)
                    request_param = query_params.get("request")
                    if request_param:
                        try:
                            decoded_param = unquote(request_param)
                            request_data = json.loads(decoded_param)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_data = None
                    else:
                        body = await request.body()
                        request_data = (
                            json.loads(body.decode("utf-8")) if body else None
                        )

                    if request_data:
                        request_id = request_data.get("id")
                        result = await _handle_mcp_request(request_data, request_id)
                        if result is not None:
                            yield f"data: {json.dumps(result)}\n\n"
                    else:
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32700,
                                "message": "Parse error: Invalid JSON in request",
                            },
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"

                    while True:
                        yield 'data: {"type":"heartbeat"}\n\n'
                        import asyncio

                        await asyncio.sleep(30)
                except Exception as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}",
                        },
                    }
                    yield f"data: {json.dumps(error_response)}\n\n"

            return StreamingResponse(
                streamable_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Handle regular GET request (e.g., for listing tools)
        try:
            # Try to get request data from query parameters
            query_params = dict(request.query_params)
            request_param = query_params.get("request")

            if request_param:
                try:
                    request_data = json.loads(request_param)
                    request_id = request_data.get("id")
                    result = await _handle_mcp_request(request_data, request_id)
                    if result is not None:
                        return Response(
                            content=json.dumps(result), media_type="application/json"
                        )
                except json.JSONDecodeError:
                    pass

            # Default response for GET without request parameter
            return Response(
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request: No request parameter provided",
                        },
                    }
                ),
                media_type="application/json",
            )
        except Exception as e:
            return Response(
                content=json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                    }
                ),
                media_type="application/json",
            )

    # Create a raw ASGI endpoint that bypasses FastAPI validation
    class RawMCPEndpoint:
        async def __call__(self, scope, receive, send):
            """Raw ASGI handler that bypasses all FastAPI validation."""
            if scope["type"] != "http":
                await send({"type": "http.response.start", "status": 404})
                await send({"type": "http.response.body", "body": b"Not Found"})
                return

            # Extract headers
            headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
            x_api_key = headers.get("x-api-key") or headers.get("authorization")

            # Check authentication
            try:
                _check_auth(x_api_key)
            except Exception:
                await send({"type": "http.response.start", "status": 401})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return

            # Check if this is a StreamableHTTP connection
            content_type = headers.get("content-type", "")
            accept = headers.get("accept", "")
            user_agent = headers.get("user-agent", "")

            # Handle StreamableHTTP connection
            if (
                "text/event-stream" in accept
                or "streamable-http" in content_type
                or "streamable" in user_agent.lower()
            ):
                # For StreamableHTTP, we need to handle it differently
                # The inspector will send JSON-RPC requests via POST to the same endpoint
                # Let's check if this is a POST request with JSON data
                method = scope.get("method", "GET")

                if method == "POST":
                    # Handle JSON-RPC request via POST for StreamableHTTP
                    try:
                        # Read the full request body
                        body_parts = []
                        while True:
                            message = await receive()
                            if message["type"] == "http.request":
                                body_parts.append(message.get("body", b""))
                                if not message.get("more_body", False):
                                    break

                        body_data = b"".join(body_parts)
                        if body_data:
                            try:
                                request_body = json.loads(body_data.decode("utf-8"))
                                request_id = request_body.get("id")
                                result = await _handle_mcp_request(
                                    request_body, request_id
                                )

                                if result is not None:
                                    response_body = json.dumps(result).encode()
                                else:
                                    # Notification, return empty success
                                    response_body = b""

                                await send(
                                    {
                                        "type": "http.response.start",
                                        "status": 200,
                                        "headers": [
                                            [b"content-type", b"application/json"]
                                        ],
                                    }
                                )
                                await send(
                                    {
                                        "type": "http.response.body",
                                        "body": response_body,
                                    }
                                )
                                return
                            except json.JSONDecodeError:
                                error_response = json.dumps(
                                    {
                                        "jsonrpc": "2.0",
                                        "id": 0,
                                        "error": {
                                            "code": -32700,
                                            "message": "Parse error: Invalid JSON",
                                        },
                                    }
                                ).encode()

                                await send(
                                    {
                                        "type": "http.response.start",
                                        "status": 200,
                                        "headers": [
                                            [b"content-type", b"application/json"]
                                        ],
                                    }
                                )
                                await send(
                                    {
                                        "type": "http.response.body",
                                        "body": error_response,
                                    }
                                )
                                return
                    except Exception as e:
                        error_response = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": 0,
                                "error": {
                                    "code": -32603,
                                    "message": f"Internal error: {str(e)}",
                                },
                            }
                        ).encode()

                        await send(
                            {
                                "type": "http.response.start",
                                "status": 200,
                                "headers": [[b"content-type", b"application/json"]],
                            }
                        )
                        await send(
                            {
                                "type": "http.response.body",
                                "body": error_response,
                            }
                        )
                        return

                # GET request for StreamableHTTP - establish SSE connection
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            [b"content-type", b"text/event-stream"],
                            [b"cache-control", b"no-cache"],
                            [b"connection", b"keep-alive"],
                            [b"access-control-allow-origin", b"*"],
                            [b"access-control-allow-headers", b"*"],
                        ],
                    }
                )

                # Send initial connection event and keep alive
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'data: {"type":"connection_established"}\n\n',
                    }
                )

                # Keep connection alive with heartbeats
                import asyncio

                try:
                    while True:
                        await asyncio.sleep(30)
                        await send(
                            {"type": "http.response.body", "body": b": heartbeat\n\n"}
                        )
                except Exception:
                    pass
                return

            # Handle regular JSON-RPC request
            try:
                # Get query string and parse it
                query_string = scope.get("query_string", b"").decode()
                query_params = {}
                if query_string:
                    for pair in query_string.split("&"):
                        if "=" in pair:
                            key, value = pair.split("=", 1)
                            query_params[key] = value

                request_param = query_params.get("request")

                if request_param:
                    try:
                        decoded_param = unquote(request_param)
                        request_body = json.loads(decoded_param)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        request_body = None
                else:
                    # Try to get from body
                    body = await receive()
                    body_data = body.get("body", b"")
                    if body_data:
                        try:
                            request_body = json.loads(body_data.decode("utf-8"))
                        except json.JSONDecodeError:
                            request_body = None
                    else:
                        request_body = None

                if request_body:
                    request_id = request_body.get("id")
                    result = await _handle_mcp_request(request_body, request_id)
                    if result is not None:
                        response_body = json.dumps(result).encode()
                    else:
                        response_body = b""
                else:
                    response_body = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32700,
                                "message": "Parse error: No valid request data found",
                            },
                        }
                    ).encode()

                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [[b"content-type", b"application/json"]],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": response_body,
                    }
                )

            except Exception as e:
                error_response = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
                    }
                ).encode()

                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [[b"content-type", b"application/json"]],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": error_response,
                    }
                )

    # Add the raw ASGI endpoint directly to the app
    app.router.add_route(
        "/mcp",
        RawMCPEndpoint(),
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    )
    if prefix != "/mcp":
        app.router.add_route(
            prefix,
            RawMCPEndpoint(),
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        )

    # Mount router
    app.include_router(router, prefix=prefix)
    return app
