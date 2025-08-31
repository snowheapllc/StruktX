from __future__ import annotations

"""
Programmatic FastAgent integration for StruktX.

This module provides a small utility to:
 - Build a FastAgent configuration at runtime (including resources if present)
 - Define agents dynamically with instruction/system prompts and model selection
 - Run agents with back-and-forth messaging

Notes:
 - Requires 'fast-agent-mcp' installed in the environment.
 - Does not start servers; it only builds client-side config and agents. For MCP
   servers, provide URLs via 'mcp_urls' or rely on configuration files in CWD.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .runtime_fast_agent import build_internal_fast_agent_config


class FastAgentManager:
    def __init__(
        self,
        *,
        init_path: str | os.PathLike[str],
        mcp_urls: Optional[List[str]] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self._init_path = str(init_path)
        self._config_dict = build_internal_fast_agent_config(
            init_path=init_path, mcp_urls=mcp_urls, default_model=default_model
        )
        self._config_file: Optional[str] = None

    def _ensure_config_file(self) -> str:
        if self._config_file:
            return self._config_file
        # Write a transient config file
        tmp_dir = tempfile.mkdtemp(prefix="struktx_fastagent_")
        cfg_path = Path(tmp_dir) / "fastagent.config.yaml"
        # Minimal YAML writer to avoid extra deps
        try:
            import yaml  # type: ignore
        except Exception:
            # Fallback to JSON superset (fast-agent accepts YAML; JSON is valid YAML)
            cfg_path.write_text(json.dumps(self._config_dict, indent=2))
        else:
            cfg_path.write_text(yaml.safe_dump(self._config_dict, sort_keys=False))
        self._config_file = str(cfg_path)
        return self._config_file

    async def run_agent(
        self,
        *,
        instruction: str | None = None,
        system_prompt: Dict[str, Any] | None = None,
        model: Optional[str] = None,
        servers: Optional[Iterable[str]] = None,
        message: str | Dict[str, Any] | List[Dict[str, Any]] = "",
        use_history: bool = True,
    ) -> str:
        """Create a temporary agent and send a message.

        - instruction: base system instruction for the agent
        - system_prompt: MCP-style PromptMessage dict to prepend (role/system)
        - model: override model string for this agent
        - servers: server names configured in the generated config
        - message: str or MCP PromptMessage/PromptMessageMultipart list
        - use_history: whether the agent maintains history for the session
        """
        try:
            from mcp_agent.core.fastagent import FastAgent  # type: ignore
            from mcp_agent.core.request_params import RequestParams  # type: ignore
            from mcp_agent.core.prompt import Prompt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "fast-agent-mcp is required to use FastAgentManager. Install 'fast-agent-mcp'."
            ) from exc

        cfg_path = self._ensure_config_file()
        fast = FastAgent("StruktX", parse_cli_args=False, config_path=cfg_path)

        agent_kwargs: Dict[str, Any] = {
            "name": "default",
            "instruction": instruction or "You are a helpful AI Agent.",
            "use_history": use_history,
        }
        if model:
            agent_kwargs["model"] = model
        if servers:
            agent_kwargs["servers"] = list(servers)

        @fast.agent(**agent_kwargs)
        async def _entrypoint():
            pass

        async with fast.run() as app:
            # Pre-apply system prompt if provided
            messages: List[Any] = []
            if system_prompt:
                messages.append(system_prompt)

            # Normalize message
            if isinstance(message, str):
                if message:
                    messages.append(Prompt.user(message))
            elif isinstance(message, dict):
                messages.append(message)
            elif isinstance(message, list):
                messages.extend(message)

            if messages:
                # Prefer generate to preserve multipart content
                resp = await app.default.generate(messages)
                return resp.last_text() or ""
            return await app.default.send("")


