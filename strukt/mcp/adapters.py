from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    get_origin,
    get_args,
    get_type_hints,
)
import inspect
from types import UnionType

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


def _py_type_to_schema(tp: Any) -> Dict[str, Any]:
    # Handle Optional/Union by selecting the first non-None arg
    origin = get_origin(tp)
    args = get_args(tp)
    try:
        from typing import Union as TypingUnion  # type: ignore
    except Exception:  # pragma: no cover
        TypingUnion = None  # type: ignore
    if origin in (TypingUnion, UnionType):
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        return _py_type_to_schema(non_none[0] if non_none else Any)

    # Primitives
    if tp in (str,):
        return {"type": "string"}
    if tp in (int,):
        return {"type": "integer"}
    if tp in (float,):
        return {"type": "number"}
    if tp in (bool,):
        return {"type": "boolean"}

    # Collections
    if origin in (list, List):
        item_schema = _py_type_to_schema(args[0]) if args else {}
        schema: Dict[str, Any] = {"type": "array"}
        if item_schema:
            schema["items"] = item_schema
        return schema
    if origin in (dict, Dict) or tp in (dict, Dict):
        return {"type": "object"}

    # Fallback
    return {"type": "object"}


def _signature_to_schema(fn: MCPCallable) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        resolved_hints = get_type_hints(fn)
    except Exception:
        resolved_hints = {}
    props: Dict[str, Any] = {}
    required: List[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        # Only treat standard kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = resolved_hints.get(
            name, param.annotation if param.annotation is not inspect._empty else Any
        )
        props[name] = _py_type_to_schema(ann)
        if param.default is inspect._empty:
            required.append(name)
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _auto_tool_name(handler_key: str, method_name: str) -> str:
    suffix = method_name[4:] if method_name.startswith("mcp_") else method_name
    base = handler_key[:-8] if handler_key.endswith("_service") else handler_key
    return f"{base}_{suffix}"


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

        # Build explicitly configured tools (with method_name)
        if configured_tools:
            for t in configured_tools:
                if not getattr(t, "method_name", None):
                    continue  # overlay-only; apply later
                raw = _resolve_method(handler, t.method_name)
                if raw is None:
                    # If user configured a method_name but we cannot resolve it, surface a clear error
                    raise ValueError(
                        f"Configured MCP tool '{t.name}' references missing method '{t.method_name}' on handler '{key}'"
                    )
                wrapped = _make_wrapped_callable(raw)
                # Prefer provided description; else use docstring; then fallback
                desc = t.description or getattr(raw, "__doc__", None) or t.name
                if getattr(t, "usage_prompt", None):
                    desc = f"{desc}\n\nUSAGE:\n{t.usage_prompt}"
                tools[t.name] = ToolSpec(
                    name=t.name,
                    description=desc,
                    parameters_schema=(
                        t.parameters_schema
                        if t.parameters_schema
                        else _signature_to_schema(raw)
                    ),
                    required_scopes=t.required_scopes or [],
                    consent_policy=t.consent_policy
                    or getattr(handler, "mcp_consent_policy", None),
                    callable=wrapped,  # type: ignore[arg-type]
                )

        # Auto-discover mcp_* methods when present
        discovered = [
            name
            for name in dir(handler)
            if name.startswith("mcp_") and callable(getattr(handler, name))
        ]
        if discovered:
            for mname in discovered:
                fn = getattr(handler, mname)
                tool_name = _auto_tool_name(key, mname)
                if tool_name in tools:
                    # do not overwrite an explicitly defined tool of same name
                    continue
                tools[tool_name] = ToolSpec(
                    name=tool_name,
                    description=getattr(fn, "__doc__", None)
                    or f"Auto-discovered MCP method '{mname}' for '{key}'",
                    parameters_schema=_signature_to_schema(fn),
                    required_scopes=[],
                    consent_policy=(
                        mcp_config.default_consent_policy if mcp_config else None
                    ),
                    callable=_make_wrapped_callable(fn),
                )
            # proceed to potential overlays
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

        # Apply overlays (configs without method_name) to tweak description/prompts
        if configured_tools:
            for t in configured_tools:
                if getattr(t, "method_name", None):
                    continue
                if t.name in tools:
                    existing = tools[t.name]
                    desc = t.description or existing.description
                    if getattr(t, "usage_prompt", None):
                        desc = f"{desc}\n\nUSAGE:\n{t.usage_prompt}"
                    existing.description = desc
                    if t.required_scopes:
                        existing.required_scopes = t.required_scopes
                    if t.consent_policy:
                        existing.consent_policy = t.consent_policy
    return tools
