from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import unquote

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
        import json
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
    def _init_mcp() -> None:
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

    def _normalize_json_schema(schema: Any) -> Dict[str, Any]:
        """Normalize JSON schema to ensure MCP compliance while preserving correct types."""
        if not isinstance(schema, dict):
            return {"type": "object"}

        # Create a copy to avoid modifying the original
        normalized = schema.copy()

        # Handle the type field specifically
        if "type" in normalized:
            type_value = normalized["type"]
            # If type is a list/array, convert to single type
            if isinstance(type_value, list):
                # Prefer object, then string, then the first non-null type
                if "object" in type_value:
                    normalized["type"] = "object"
                elif "string" in type_value:
                    normalized["type"] = "string"
                elif "array" in type_value:
                    normalized["type"] = "array"
                elif "integer" in type_value:
                    normalized["type"] = "integer"
                elif "number" in type_value:
                    normalized["type"] = "number"
                elif "boolean" in type_value:
                    normalized["type"] = "boolean"
                elif "null" in type_value and len(type_value) > 1:
                    # Find first non-null type
                    non_null_types = [t for t in type_value if t != "null"]
                    normalized["type"] = (
                        non_null_types[0] if non_null_types else "object"
                    )
                else:
                    normalized["type"] = type_value[0] if type_value else "object"
            # Keep valid JSON Schema types as-is
            elif type_value in [
                "null",
                "boolean",
                "object",
                "array",
                "number",
                "string",
                "integer",
            ]:
                # Don't change valid types
                pass
            else:
                # Invalid type, default to object
                normalized["type"] = "object"
        else:
            # No type specified, default to object
            normalized["type"] = "object"

        # Recursively normalize nested schemas
        if "properties" in normalized and isinstance(normalized["properties"], dict):
            for prop_name, prop_schema in normalized["properties"].items():
                # Only normalize if the property schema is actually a complex schema
                # Don't normalize simple property definitions that already have valid types
                if isinstance(prop_schema, dict) and (
                    "properties" in prop_schema
                    or "items" in prop_schema
                    or "additionalProperties" in prop_schema
                ):
                    normalized["properties"][prop_name] = _normalize_json_schema(
                        prop_schema
                    )

        # Keep array items and normalize them
        if "items" in normalized:
            normalized["items"] = _normalize_json_schema(normalized["items"])

        if "additionalProperties" in normalized and isinstance(
            normalized["additionalProperties"], dict
        ):
            normalized["additionalProperties"] = _normalize_json_schema(
                normalized["additionalProperties"]
            )

        return normalized

    def _handle_mcp_request(
        request_body: Dict[str, Any], request_id: Any
    ) -> Dict[str, Any]:
        """Handle MCP JSON-RPC request logic."""
        # Validate JSON-RPC request structure
        if not isinstance(request_body, dict):
            return {
                "jsonrpc": "2.0",
                "id": 0,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: body must be an object",
                },
            }

        jsonrpc = request_body.get("jsonrpc")
        method = request_body.get("method")
        params = request_body.get("params")

        if jsonrpc != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: jsonrpc must be '2.0'",
                },
            }

        if not method:
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: method is required",
                },
            }

        # Handle MCP initialization
        if method == "initialize":
            try:
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
                            "name": cfg.mcp.server_name or "strukt-mcp-server",
                            "version": "1.0.0",
                        },
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }

        # Handle initialized notification (no response needed)
        elif method == "notifications/initialized":
            # This is a notification, no response needed
            return None

        elif method == "tools/list":
            try:
                tools_raw = mcp.list_tools()
                # Debug: Print the raw tools to see what we're working with
                print(f"DEBUG: Found {len(tools_raw)} raw tools")
                for i, tool in enumerate(tools_raw):
                    print(
                        f"DEBUG: Tool {i}: {tool.get('name', 'unnamed')} - outputSchema: {tool.get('outputSchema', 'none')}"
                    )

                # Fix outputSchema to ensure proper JSON schema format
                tools = []
                for i, tool in enumerate(tools_raw):
                    tool_copy = tool.copy()

                    # Remove outputSchema completely if it's problematic, or normalize it
                    if "outputSchema" in tool_copy:
                        output_schema = tool_copy["outputSchema"]
                        if output_schema:
                            try:
                                # Normalize the schema to ensure MCP compliance
                                normalized_schema = _normalize_json_schema(
                                    output_schema
                                )
                                print(
                                    f"DEBUG: Tool {i} normalized schema: {normalized_schema}"
                                )
                                tool_copy["outputSchema"] = normalized_schema
                            except Exception as e:
                                # If normalization fails, remove outputSchema entirely
                                print(
                                    f"WARNING: Failed to normalize schema for tool {i} ({tool.get('name', 'unknown')}): {e}"
                                )
                                print(f"Original schema: {output_schema}")
                                del tool_copy["outputSchema"]
                        else:
                            # Remove empty outputSchema
                            del tool_copy["outputSchema"]

                    tools.append(tool_copy)

                # Final debug output
                print(f"DEBUG: Returning {len(tools)} tools")
                for i, tool in enumerate(tools):
                    print(
                        f"DEBUG: Final tool {i}: {tool.get('name', 'unnamed')} - has outputSchema: {'outputSchema' in tool}"
                    )

                result = {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": {"tools": tools},
                }

                # Debug: Print the complete JSON response
                import json as json_module

                print(
                    f"DEBUG: Complete tools/list response: {json_module.dumps(result, indent=2)}"
                )

                return result
            except Exception as e:
                print(f"ERROR in tools/list: {e}")
                import traceback

                traceback.print_exc()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }

        elif method == "tools/call":
            if not params:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32602, "message": "Missing params"},
                }

            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32602, "message": "Missing tool name"},
                }

            try:
                result = mcp.call_tool(tool_name=tool_name, args=arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": result,
                }
            except ValueError as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32602, "message": str(e)},
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }

        # Handle prompts
        elif method == "prompts/list":
            try:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": {
                        "prompts": []  # No prompts implemented yet
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }

        elif method == "prompts/get":
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {"code": -32602, "message": "No prompts available"},
            }

        # Handle resources
        elif method == "resources/list":
            try:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "result": {
                        "resources": []  # No resources implemented yet
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else 0,
                    "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
                }

        elif method == "resources/read":
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {"code": -32602, "message": "No resources available"},
            }

        # Handle sampling
        elif method == "sampling/createMessage":
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {"code": -32601, "message": "Sampling not implemented"},
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id if request_id is not None else 0,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
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
                        result = _handle_mcp_request(request_data, request_id)
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
                    result = _handle_mcp_request(request_data, request_id)
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
                                result = _handle_mcp_request(request_body, request_id)

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
                    result = _handle_mcp_request(request_body, request_id)
                    response_body = json.dumps(result).encode()
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
