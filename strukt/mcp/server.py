from __future__ import annotations

import json
from typing import Any, Dict, List

from ..interfaces import Handler, MemoryEngine
from ..logging import get_logger
from .adapters import ToolSpec, build_tools_from_handlers, _normalize_schema_for_mcp
from .auth import APIKeyAuthorizer, APIKeyAuthConfig
from .permissions import ConsentPolicy, ConsentDecision, ConsentStore

_log = get_logger("mcp.server")


class MCPServerApp:
    """Runtime-agnostic MCP server surface.

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
        # Build tools with default consent policy from config provided by host later
        self._tools: Dict[str, ToolSpec] = build_tools_from_handlers(
            handlers=handlers, include=include_handlers, mcp_config=None
        )
        self._consent = ConsentStore(memory)
        self._authz = APIKeyAuthorizer(api_key_auth)

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
                    normalized_schema = _normalize_schema_for_mcp(t.output_schema)
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

    # Hosting layer invokes this to run the tool after auth+consent
    def call_tool(self, *, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        spec = self._tools.get(tool_name)
        if spec and spec.callable:
            try:
                result = spec.callable(**args)

                # Debug: Log what we got from the tool
                print(f"DEBUG call_tool: {tool_name} returned type: {type(result)}")
                print(f"DEBUG call_tool: {tool_name} result: {result}")

                # Check if this tool has an output schema (indicating structured output expected)
                has_output_schema = (
                    hasattr(spec, "output_schema") and spec.output_schema
                )
                print(
                    f"DEBUG call_tool: {tool_name} has_output_schema: {has_output_schema}"
                )

                # If result is already in MCP format, return as-is
                if isinstance(result, dict) and "content" in result:
                    print(
                        f"DEBUG call_tool: {tool_name} returning pre-formatted MCP result"
                    )
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
                            result
                            if isinstance(result, dict)
                            else {"data": list(result)}
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
                        text_repr = json.dumps(
                            structured_result, indent=2, ensure_ascii=False
                        )
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

            except Exception as e:
                # Return error in MCP format
                return {
                    "content": [
                        {"type": "text", "text": f"Tool execution failed: {str(e)}"}
                    ],
                    "isError": True,
                }
        raise ValueError(f"Unknown tool: {tool_name}")
