"""
Fast-agent runtime adapter for StruktX MCP.

This module is optional. It requires fast-agent to be installed in the
environment. The adapter exposes a small class that a fast-agent server can
instantiate to list tools and dispatch calls into MCPServerApp.

Docs: https://fast-agent.ai/
"""

from __future__ import annotations
from .server import MCPServerApp
from typing import Any, Dict, List, Optional


class FastAgentAdapter:
    def __init__(self, app: MCPServerApp) -> None:
        self._app = app

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._app.list_tools()

    def authorize(self, headers: dict[str, str]) -> bool:
        return self._app.check_api_key(headers)

    def call_tool(self, *, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self._app.call_tool(tool_name=tool_name, args=args)

    def evaluate_consent(
        self, *, user_id: str, tool_name: str, policy: Optional[str]
    ) -> bool:
        return self._app.evaluate_consent(
            user_id=user_id, tool_name=tool_name, policy=policy or "ask-once"
        )

    def record_consent(self, *, user_id: str, tool_name: str, decision: str) -> None:
        self._app.record_consent(
            user_id=user_id, tool_name=tool_name, decision=decision
        )
