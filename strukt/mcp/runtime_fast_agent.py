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
import os
from pathlib import Path


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


def _find_resources_dir(start_path: str | os.PathLike[str]) -> Optional[Path]:
    """Search upward from start_path for a 'strukt_mcp_resources' directory."""
    try:
        p = Path(start_path).resolve()
    except Exception:
        return None
    for current in [p] + list(p.parents):
        candidate = current / "strukt_mcp_resources"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def build_internal_fast_agent_config(
    *,
    init_path: str | os.PathLike[str],
    mcp_urls: List[str] | None = None,
    default_model: str | None = None,
) -> Dict[str, Any]:
    """Construct a fast-agent compatible config dict programmatically.

    - Finds and mounts 'strukt_mcp_resources' if present; otherwise leaves resources disabled.
    - Optionally sets default model.
    - Optionally wires provided MCP server URLs.
    """
    cfg: Dict[str, Any] = {}
    if default_model:
        cfg["default_model"] = default_model

    resources_dir = _find_resources_dir(init_path)
    servers: Dict[str, Any] = {}

    # Attach URL-based servers if any
    if mcp_urls:
        for idx, u in enumerate(mcp_urls):
            name = f"url_{idx}"
            transport = "sse" if u.endswith("/sse") else "http"
            servers[name] = {
                "transport": transport,
                "url": u,
            }

    # Attach prompt-server only if resources dir exists
    if resources_dir is not None:
        servers["prompts"] = {
            "transport": "stdio",
            "command": "prompt-server",
            "args": [str(resources_dir)],
        }

    if servers:
        cfg["mcp"] = {"servers": servers}

    # Elicitation: disabled per requirements
    cfg["logger"] = {"progress_display": False}
    return cfg
