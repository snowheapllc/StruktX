from __future__ import annotations

from typing import Any, Dict, List

from ..interfaces import Handler, MemoryEngine
from ..logging import get_logger
from .adapters import ToolSpec, PromptSpec, build_tools_from_handlers, build_prompts_from_handlers
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
        # Build prompts by discovery (mcp_prompt_* methods)
        self._prompts: Dict[str, PromptSpec] = build_prompts_from_handlers(
            handlers=handlers, include=include_handlers
        )
        self._consent = ConsentStore(memory)
        self._authz = APIKeyAuthorizer(api_key_auth)

    # Hosting layer should call this to list tools with metadata
    def list_tools(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for t in self._tools.values():
            tools.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters_schema,
                    "x-scopes": t.required_scopes,
                    "x-consent-policy": t.consent_policy,
                }
            )
        # Append prompts as MCP Prompts entries
        for p in self._prompts.values():
            tools.append(
                {
                    "name": p.name,
                    "description": p.description,
                    "inputSchema": p.arguments_schema,
                    "x-type": "prompt",
                }
            )
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
        # First try tool invocation
        spec = self._tools.get(tool_name)
        if spec and spec.callable:
            return spec.callable(**args)
        # Then try prompt invocation
        pspec = self._prompts.get(tool_name)
        if pspec and pspec.callable:
            result = pspec.callable(**args)
            # Normalize result to MCP-friendly dict
            if isinstance(result, list):
                return {"content": result}
            if isinstance(result, dict):
                return {"content": [result]}
            # Fallback textual content
            return {"content": [{"type": "text", "text": str(result)}]}
        raise ValueError(f"Unknown tool or prompt: {tool_name}")
