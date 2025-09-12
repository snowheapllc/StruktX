from __future__ import annotations

import json
from typing import Any, Dict, List

from mcp.server import Server
from mcp.types import Tool, TextContent

from ..interfaces import Handler, MemoryEngine
from ..logging import get_logger
from .adapters import ToolSpec, build_tools_from_handlers
from .auth import APIKeyAuthorizer, APIKeyAuthConfig
from .permissions import ConsentPolicy, ConsentDecision, ConsentStore

_log = get_logger("mcp.server")


class MCPServerApp:
    """Runtime-agnostic MCP server surface using official MCP SDK.

    A hosting layer (e.g., fast-agent) can bind this app to stdio or HTTP.
    """

    def __init__(
        self,
        *,
        server_name: str,
        handlers: Dict[str, Handler],
        include_handlers: List[str],
        memory: MemoryEngine | None,
        api_key_auth: APIKeyAuthConfig,
    ) -> None:
        self.server_name = server_name
        self._handlers = handlers
        self._include_handlers = include_handlers
        self._memory = memory
        self._api_key_auth = api_key_auth

        # Initialize official MCP server
        self._server = Server(server_name)

        # Build tools with default consent policy from config provided by host later
        self._tools: Dict[str, ToolSpec] = build_tools_from_handlers(
            handlers=handlers, include=include_handlers, mcp_config=None
        )
        self._consent = ConsentStore(memory)
        self._authz = APIKeyAuthorizer(api_key_auth)

        # Register tools with the official MCP server
        self._register_tools_with_server()

    def _register_tools_with_server(self) -> None:
        """Register all tools with the official MCP server."""

        # Register list_tools handler
        @self._server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            tools = []
            for tool_name, tool_spec in self._tools.items():
                tools.append(
                    Tool(
                        name=tool_name,
                        description=tool_spec.description,
                        inputSchema=tool_spec.input_schema,
                    )
                )
            return tools

        # Register call_tool handler
        @self._server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name in self._tools:
                try:
                    result = self._call_tool_sync(name, arguments)
                    formatted_result = self._format_tool_result(
                        name, self._tools[name], result
                    )

                    # Extract text content from formatted result
                    if (
                        isinstance(formatted_result, dict)
                        and "content" in formatted_result
                    ):
                        content_items = formatted_result["content"]
                        if content_items and isinstance(content_items[0], dict):
                            text_content = content_items[0].get("text", str(result))
                        else:
                            text_content = str(result)
                    else:
                        text_content = str(result)

                    return [TextContent(type="text", text=text_content)]
                except Exception as e:
                    return [TextContent(type="text", text=f"Error: {str(e)}")]
            else:
                raise ValueError(f"Unknown tool: {name}")

    @property
    def server(self) -> Server:
        """Get the official MCP server instance."""
        return self._server

    # Hosting layer should call this to list tools with metadata
    def list_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for t in self._tools.values():
            tool_spec = {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.parameters_schema,
            }
            # Only include title if it's different from name
            if hasattr(t, "title") and t.title and t.title != t.name:
                tool_spec["title"] = t.title
            # Only include outputSchema if provided, and normalize it for MCP compliance
            if hasattr(t, "output_schema") and t.output_schema:
                try:
                    normalized_schema = t.output_schema
                    tool_spec["outputSchema"] = normalized_schema
                except Exception as e:
                    _log.warn(
                        f"Failed to normalize output schema for tool {t.name}: {e}"
                    )
                    # Skip outputSchema if normalization fails
                    pass
            tools.append(tool_spec)
        return tools

    def check_api_key(self, headers: dict[str, str]) -> bool:
        return self._authz.is_authorized(headers)

    def evaluate_consent(
        self, *, user_id: str, tool_name: str, policy: str | None
    ) -> bool:
        if policy == ConsentPolicy.NEVER_ALLOW:
            return False
        if policy == ConsentPolicy.ALWAYS_ALLOW:
            return True
        # ASK_ONCE or ALWAYS_ASK
        remembered = self._consent.get(user_id, tool_name)
        if remembered == ConsentPolicy.NEVER_ALLOW:
            return False
        if remembered == ConsentPolicy.ALWAYS_ALLOW:
            return True
        # Default to require consent UI by hosting layer for ALWAYS_ASK / unset
        return False

    def record_consent(self, *, user_id: str, tool_name: str, decision: str) -> None:
        self._consent.set(
            ConsentDecision(user_id=user_id, tool_name=tool_name, decision=decision)
        )

    async def call_tool(
        self,
        *,
        name: str,
        arguments: Dict[str, Any],
        api_key: str | None = None,
        mcp_config: ConsentPolicy | None = None,
    ) -> Any:
        """Call a tool by name with arguments.

        Args:
            name: Tool name
            arguments: Tool arguments
            api_key: API key for authorization (required if auth is enabled)
            mcp_config: Consent policy for this call

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or arguments invalid
            PermissionError: If authorization fails
        """
        # Validate API key - now enforced for all requests
        if not api_key:
            raise PermissionError("API key is required for all MCP requests")

        if not self._authz.is_authorized({self._api_key_auth.header_name: api_key}):
            raise PermissionError("Invalid API key")

        # Check consent if policy is provided
        if mcp_config and mcp_config.require_consent:
            decision = await self._consent.get_consent(
                tool_name=name, arguments=arguments, policy=mcp_config
            )
            if decision != ConsentDecision.ALLOW:
                raise PermissionError(f"Consent denied for tool '{name}'")

        # Use the synchronous tool calling logic
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found")

        tool_spec = self._tools[name]
        handler = self._handlers[tool_spec.handler_name]

        # Validate arguments against schema
        try:
            # The handler will validate arguments internally
            result = await handler.handle(arguments)
            _log.info(f"Tool '{name}' executed successfully")
            return result
        except Exception as e:
            _log.error(f"Tool '{name}' execution failed: {e}")
            raise

    def _format_tool_result(
        self, tool_name: str, spec: ToolSpec, result: Any
    ) -> Dict[str, Any]:
        """Format tool result according to MCP standards."""
        # Debug: Log what we got from the tool
        print(f"DEBUG call_tool: {tool_name} returned type: {type(result)}")
        print(f"DEBUG call_tool: {tool_name} result: {result}")

        # Check if this tool has an output schema (indicating structured output expected)
        has_output_schema = hasattr(spec, "output_schema") and spec.output_schema
        print(f"DEBUG call_tool: {tool_name} has_output_schema: {has_output_schema}")

        # If result is already in MCP format, return as-is
        if isinstance(result, dict) and "content" in result:
            print(f"DEBUG call_tool: {tool_name} returning pre-formatted MCP result")
            return result

        # For tools with output schema, return structured data
        if has_output_schema:
            print(f"DEBUG call_tool: {tool_name} processing structured output")
            # Extract structured data
            if hasattr(result, "model_dump"):
                structured_result = result.model_dump()
                print(
                    f"DEBUG call_tool: {tool_name} Pydantic model_dump: {structured_result}"
                )
            elif isinstance(result, (dict, list, tuple)):
                structured_result = (
                    result if isinstance(result, dict) else {"data": list(result)}
                )
                print(
                    f"DEBUG call_tool: {tool_name} dict/list result: {structured_result}"
                )
            else:
                structured_result = {"value": result}
                print(
                    f"DEBUG call_tool: {tool_name} other type wrapped: {structured_result}"
                )

            # For MCP structured content, use structuredContent field
            # Also include JSON text representation for backwards compatibility
            try:
                text_repr = json.dumps(structured_result, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                text_repr = str(structured_result)
            content = [{"type": "text", "text": text_repr}]

            # Return the response with proper MCP structure
            final_result = {
                "content": content,
                "isError": False,
                "structuredContent": structured_result,
                # Also add structured data at root level for validation
                **structured_result,
                # Provide a 'data' wrapper for clients expecting data.* paths
                "data": structured_result,
            }
            print(
                f"DEBUG call_tool: {tool_name} final structured result: {final_result}"
            )
            return final_result

        # For tools without output schema, return as text content
        print(f"DEBUG call_tool: {tool_name} returning as text content")
        return {
            "content": [{"type": "text", "text": str(result)}],
            "isError": False,
        }

    def _call_tool_sync(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Synchronous tool calling logic."""
        spec = self._tools.get(tool_name)
        if spec and spec.callable:
            try:
                return spec.callable(**args)
            except Exception as e:
                raise ValueError(f"Tool execution failed: {str(e)}")
        raise ValueError(f"Unknown tool: {tool_name}")
