"""
Handler adapters for FastMCP v2 integration.

This module provides utilities to discover and convert Strukt handlers with
mcp_* prefixed methods into FastMCP tools.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    get_origin,
    get_args,
    get_type_hints,
)
from types import UnionType

from ..interfaces import Handler
from ..logging import get_logger

_log = get_logger("mcp_v2.handler_adapters")


@dataclass
class ToolSpec:
    """Specification for a FastMCP tool derived from a handler method."""

    name: str
    description: str
    method: Callable[..., Any]
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


def _py_type_to_schema(tp: Any) -> Dict[str, Any]:
    """Convert Python type to JSON schema, adapted from v1 adapters."""
    # Handle Pydantic models
    if hasattr(tp, "model_json_schema"):
        try:
            pydantic_schema = tp.model_json_schema()
            return pydantic_schema
        except Exception:
            pass

    # Handle dataclasses
    if hasattr(tp, "__dataclass_fields__"):
        try:
            properties = {}
            required = []
            for field_name, field_info in tp.__dataclass_fields__.items():
                field_type = field_info.type
                field_schema = _py_type_to_schema(field_type)
                properties[field_name] = field_schema
                # Check if field has a default value
                if field_info.default == field_info.default_factory == inspect._empty:
                    required.append(field_name)

            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required
            return schema
        except Exception as e:
            _log.debug(f"Exception in dataclass handling: {e}")
            pass

    # Handle basic types - check string representation since type objects may differ
    type_str = str(tp)
    if type_str in ["<class 'str'>", "str"] or (
        hasattr(tp, "__name__") and tp.__name__ == "str"
    ):
        return {"type": "string"}
    elif type_str in ["<class 'int'>", "int"] or (
        hasattr(tp, "__name__") and tp.__name__ == "int"
    ):
        return {"type": "integer"}
    elif type_str in ["<class 'float'>", "float"] or (
        hasattr(tp, "__name__") and tp.__name__ == "float"
    ):
        return {"type": "number"}
    elif type_str in ["<class 'bool'>", "bool"] or (
        hasattr(tp, "__name__") and tp.__name__ == "bool"
    ):
        return {"type": "boolean"}
    elif type_str in ["<class 'list'>", "list"] or (
        hasattr(tp, "__name__") and tp.__name__ == "list"
    ):
        return {"type": "array"}
    elif type_str in ["<class 'dict'>", "dict"] or (
        hasattr(tp, "__name__") and tp.__name__ == "dict"
    ):
        return {"type": "object"}

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

    # Primitives - only return MCP-supported types
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
        item_schema = _py_type_to_schema(args[0]) if args else {"type": "object"}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Dict) or tp in (dict, Dict):
        return {"type": "object"}

    # Fallback - always return a valid MCP type
    return {"type": "object"}


def _signature_to_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    """Extract JSON schema from function signature, adapted from v1 adapters."""
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


def _extract_output_schema(fn: Callable[..., Any]) -> Dict[str, Any] | None:
    """Extract output schema from function return type annotation."""
    try:
        resolved_hints = get_type_hints(fn)
        return_type = resolved_hints.get("return")
        if return_type is None or return_type == inspect._empty:
            return None
        schema = _py_type_to_schema(return_type)
        return schema
    except Exception:
        return None


def _extract_docstring(
    method: Callable[..., Any], method_name: str, handler_key: str
) -> str:
    """Extract and clean up docstring from a method."""
    # Try to get docstring from the method
    docstring = getattr(method, "__doc__", None)

    _log.info(
        f"Extracting docstring for {handler_key}.{method_name}: docstring={docstring}"
    )

    if docstring:
        # Clean up the docstring - remove leading/trailing whitespace and normalize
        cleaned = docstring.strip()
        if cleaned:
            # Take the first line if it's a multi-line docstring
            first_line = cleaned.split("\n")[0].strip()
            result = first_line if first_line else f"Tool from handler '{handler_key}'"
            _log.info(f"Extracted docstring: '{result}'")
            return result

    # Fallback to a more descriptive default
    fallback = f"Tool from handler '{handler_key}'"
    _log.info(f"Using fallback docstring: '{fallback}'")
    return fallback


def _auto_tool_name(handler_key: str, method_name: str) -> str:
    """Generate tool name from handler key and method name."""
    suffix = method_name[4:] if method_name.startswith("mcp_") else method_name
    base = handler_key[:-8] if handler_key.endswith("_service") else handler_key
    return f"{base}_{suffix}"


def discover_mcp_methods(handler: Handler, handler_key: str) -> List[ToolSpec]:
    """Discover all mcp_* prefixed methods on a handler and convert to ToolSpecs."""
    tools: List[ToolSpec] = []

    # Find all methods starting with mcp_
    mcp_methods = [
        name
        for name in dir(handler)
        if name.startswith("mcp_") and callable(getattr(handler, name))
    ]

    for method_name in mcp_methods:
        method = getattr(handler, method_name)
        tool_name = _auto_tool_name(handler_key, method_name)

        # Extract description from docstring
        description = _extract_docstring(method, method_name, handler_key)

        # Generate input schema from method signature
        input_schema = _signature_to_schema(method)

        # Extract output schema from return type annotation
        output_schema = _extract_output_schema(method)

        tool_spec = ToolSpec(
            name=tool_name,
            description=description,
            method=method,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        tools.append(tool_spec)
        _log.info(
            f"Discovered MCP tool '{tool_name}' from method '{method_name}' on handler '{handler_key}'"
        )

    return tools


def build_fastmcp_tool_from_method(tool_spec: ToolSpec, fastmcp_instance: Any) -> Any:
    """Convert a ToolSpec to a FastMCP-compatible tool using the @mcp.tool decorator."""

    # Get the original method's signature to preserve parameter types
    import inspect

    original_sig = inspect.signature(tool_spec.method)

    # Create a wrapper function that preserves the original signature
    def tool_wrapper(*args, **kwargs) -> Any:
        """Wrapper function for FastMCP tool."""
        try:
            # Bind arguments to the original method signature
            bound_args = original_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            result = tool_spec.method(*bound_args.args, **bound_args.kwargs)
            _log.debug(f"Tool '{tool_spec.name}' executed successfully")
            return result
        except Exception as e:
            _log.error(f"Tool '{tool_spec.name}' execution failed: {e}")
            raise

    # Preserve the original method's signature and metadata
    tool_wrapper.__name__ = tool_spec.name
    tool_wrapper.__doc__ = tool_spec.description
    tool_wrapper.__signature__ = original_sig
    tool_wrapper.__annotations__ = tool_spec.method.__annotations__

    # Use the @mcp.tool decorator to create a proper FunctionTool
    return fastmcp_instance.tool(
        name=tool_spec.name, description=tool_spec.description
    )(tool_wrapper)


def register_handler_tools(
    fastmcp_instance: Any,
    handler: Handler,
    handler_key: str,
    include_handlers: List[str] | None = None,
) -> List[tuple[str, str]]:
    """Register all mcp_* tools from a handler with a FastMCP instance.

    Args:
        fastmcp_instance: The FastMCP instance to register tools with
        handler: The Strukt handler to discover tools from
        handler_key: The key/name of the handler
        include_handlers: Optional list of handler keys to include (if None, include all)

    Returns:
        List of tuples (tool_name, description)
    """
    if include_handlers is not None and handler_key not in include_handlers:
        _log.debug(f"Skipping handler '{handler_key}' - not in include list")
        return []

    tool_specs = discover_mcp_methods(handler, handler_key)
    registered_tools: List[tuple[str, str]] = []

    for tool_spec in tool_specs:
        try:
            # Build FastMCP tool from method using the @mcp.tool decorator
            build_fastmcp_tool_from_method(tool_spec, fastmcp_instance)

            # The tool is automatically registered when created with @mcp.tool decorator
            # No need to call add_tool separately

            registered_tools.append((tool_spec.name, tool_spec.description))
            _log.info(
                f"Registered FastMCP tool '{tool_spec.name}' from handler '{handler_key}'"
            )

        except Exception as e:
            _log.error(
                f"Failed to register tool '{tool_spec.name}' from handler '{handler_key}': {e}"
            )

    return registered_tools
