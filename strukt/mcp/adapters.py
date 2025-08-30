from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..interfaces import Handler
from ..config import MCPConfig, MCPToolConfig
from ..types import InvocationState, HandlerResult


class MCPCallable(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    required_scopes: List[str] = field(default_factory=list)
    consent_policy: str | None = None  # e.g., "always-ask", "ask-once", etc.
    # The function the MCP runtime should call; provided by handler via mcp_handle
    callable: Optional[MCPCallable] = None


def _resolve_method(obj: Any, method_path: str | None) -> Optional[MCPCallable]:
    if not method_path:
        return None

    def _walk(root: Any, path: str) -> Optional[MCPCallable]:
        parts = path.split(".")
        cur: Any = root
        for p in parts:
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur if callable(cur) else None

    # try provided path
    fn = _walk(obj, method_path)
    if fn:
        return fn
    # special-case: allow "toolkit." -> "_toolkit." fallback
    if method_path.startswith("toolkit.") and hasattr(obj, "_toolkit"):
        alt = "_" + method_path
        fn = _walk(obj, alt)
        if fn:
            return fn
    return None


def _wrap_result_to_mcp_content(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return {"content": [{"type": "json", "json": result}]}
    if isinstance(result, (list, tuple)):
        return {"content": [{"type": "json", "json": result}]}
    return {"content": [{"type": "text", "text": str(result)}]}


def _make_generic_callable(bound_handler: Handler) -> MCPCallable:
    def generic_callable(
        *, text: str, context: Dict[str, Any] | None = None, **_: Any
    ) -> Dict[str, Any]:
        state = InvocationState(text=text, context=context or {})
        result: HandlerResult = bound_handler.handle(state, [text])
        return {"content": [{"type": "text", "text": result.response}]}

    return generic_callable


def _make_wrapped_callable(underlying: MCPCallable) -> MCPCallable:
    def call_wrapper(**kwargs: Any) -> Dict[str, Any]:
        return _wrap_result_to_mcp_content(underlying(**kwargs))

    return call_wrapper


def build_tools_from_handlers(
    *,
    handlers: Dict[str, Handler],
    include: List[str],
    mcp_config: MCPConfig | None = None,
) -> Dict[str, ToolSpec]:
    """Create ToolSpec mapping from selected handlers.

    Handlers are expected to expose an attribute or method `mcp_handle` returning
    a callable with keyword-only parameters matching parameters_schema.
    If absent, a minimal generic wrapper will be generated that routes to
    `handle()` using text/context inputs.
    """
    tools: Dict[str, ToolSpec] = {}
    for key, handler in handlers.items():
        if include and key not in include:
            continue
        configured_tools: List[MCPToolConfig] = []
        if mcp_config and mcp_config.tools and key in mcp_config.tools:
            configured_tools = mcp_config.tools.get(key, []) or []

        if configured_tools:
            for t in configured_tools:
                call = _resolve_method(handler, t.method_name)
                if call is None:
                    # If user configured a method_name but we cannot resolve it, surface a clear error
                    raise ValueError(
                        f"Configured MCP tool '{t.name}' references missing method '{t.method_name}' on handler '{key}'"
                    )
                call = _make_wrapped_callable(call)
                # Combine base description with optional usage prompt for richer guidance
                desc = t.description
                if getattr(t, "usage_prompt", None):
                    desc = f"{t.description}\n\nUSAGE:\n{t.usage_prompt}"
                tools[t.name] = ToolSpec(
                    name=t.name,
                    description=desc,
                    parameters_schema=t.parameters_schema
                    or {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "context": {"type": "object"},
                        },
                        "required": ["text"],
                    },
                    required_scopes=t.required_scopes or [],
                    consent_policy=t.consent_policy
                    or getattr(handler, "mcp_consent_policy", None),
                    callable=call,  # type: ignore[arg-type]
                )
        else:
            # Single default tool derived from handler
            tool_name = key
            desc = (
                getattr(handler, "mcp_description", None) or f"Strukt handler '{key}'"
            )
            schema: Dict[str, Any] = getattr(
                handler, "mcp_parameters_schema", None
            ) or {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "User input text"},
                    "context": {"type": "object", "description": "Optional context"},
                },
                "required": ["text"],
            }
            required_scopes: List[str] = (
                getattr(handler, "mcp_required_scopes", None) or []
            )
            consent = getattr(handler, "mcp_consent_policy", None) or (
                mcp_config.default_consent_policy if mcp_config else None
            )

            mcp_callable = getattr(handler, "mcp_handle", None)
            if not callable(mcp_callable):
                mcp_callable = _make_generic_callable(handler)
            tools[tool_name] = ToolSpec(
                name=tool_name,
                description=desc,
                parameters_schema=schema,
                required_scopes=required_scopes,
                consent_policy=consent,
                callable=mcp_callable,  # type: ignore[arg-type]
            )
    return tools
