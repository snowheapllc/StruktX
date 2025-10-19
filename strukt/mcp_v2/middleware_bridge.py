"""
Middleware bridge for FastMCP v2 integration.

This module provides utilities to bridge Strukt's middleware system to
FastMCP's middleware patterns.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import asyncio
import functools

from ..middleware import Middleware
from ..types import InvocationState, HandlerResult
from ..logging import get_logger

_log = get_logger("mcp_v2.middleware_bridge")


def create_fastmcp_middleware(
    strukt_middleware: List[Middleware],
) -> Callable[..., Any]:
    """Convert Strukt middleware to FastMCP middleware pattern.

    Args:
        strukt_middleware: List of Strukt middleware instances to bridge

    Returns:
        FastMCP-compatible middleware function
    """
    if not strukt_middleware:
        _log.debug("No Strukt middleware to bridge")
        return lambda handler: handler

    def fastmcp_middleware(handler: Callable[..., Any]) -> Callable[..., Any]:
        """FastMCP middleware wrapper that bridges Strukt middleware."""

        @functools.wraps(handler)
        async def middleware_wrapper(*args, **kwargs) -> Any:
            """Wrapper that applies Strukt middleware to FastMCP tool calls."""

            # Extract context from FastMCP call
            # FastMCP typically passes tool arguments as kwargs
            tool_name = kwargs.get("tool_name", "unknown")
            arguments = {k: v for k, v in kwargs.items() if k != "tool_name"}

            # Create a minimal InvocationState for middleware
            # This is a simplified representation - in practice, you might need
            # to extract more context from the FastMCP request
            state = InvocationState(
                text=str(arguments),  # Convert arguments to text representation
                context={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "source": "fastmcp",
                },
            )

            # Apply before_handle middleware
            processed_state = state
            processed_parts = [str(arguments)]

            for middleware in strukt_middleware:
                try:
                    processed_state, processed_parts = middleware.before_handle(
                        processed_state, tool_name, processed_parts
                    )
                except Exception as e:
                    _log.warning(
                        f"Error in before_handle middleware {middleware.__class__.__name__}: {e}"
                    )

            # Execute the original handler
            try:
                # Convert back to kwargs for the original handler
                handler_kwargs = {
                    k: v
                    for k, v in processed_state.context.get("arguments", {}).items()
                }
                result = (
                    await handler(**handler_kwargs)
                    if asyncio.iscoroutinefunction(handler)
                    else handler(**handler_kwargs)
                )

                # Wrap result in HandlerResult for middleware compatibility
                handler_result = HandlerResult(response=result, status="success")

            except Exception as e:
                _log.error(f"Tool execution failed: {e}")
                handler_result = HandlerResult(
                    response=f"Error: {str(e)}", status="error"
                )

            # Apply after_handle middleware
            final_result = handler_result
            for middleware in strukt_middleware:
                try:
                    final_result = middleware.after_handle(
                        processed_state, tool_name, final_result
                    )
                except Exception as e:
                    _log.warning(
                        f"Error in after_handle middleware {middleware.__class__.__name__}: {e}"
                    )

            # Return the final result
            return (
                final_result.response
                if hasattr(final_result, "response")
                else final_result
            )

        return middleware_wrapper

    _log.info(
        f"Created FastMCP middleware bridge for {len(strukt_middleware)} Strukt middleware"
    )
    return fastmcp_middleware


def create_request_middleware(
    strukt_middleware: List[Middleware],
) -> Callable[..., Any]:
    """Create FastMCP request middleware that applies before_handle hooks."""

    def request_middleware(request: Dict[str, Any]) -> Dict[str, Any]:
        """FastMCP request middleware."""
        # Extract tool information from request
        tool_name = request.get("method", "unknown")
        arguments = request.get("params", {})

        # Create InvocationState
        state = InvocationState(
            text=str(arguments),
            context={
                "tool_name": tool_name,
                "arguments": arguments,
                "source": "fastmcp_request",
            },
        )

        # Apply before_handle middleware
        processed_state = state
        processed_parts = [str(arguments)]

        for middleware in strukt_middleware:
            try:
                processed_state, processed_parts = middleware.before_handle(
                    processed_state, tool_name, processed_parts
                )
            except Exception as e:
                _log.warning(
                    f"Error in request middleware {middleware.__class__.__name__}: {e}"
                )

        # Update request with processed state
        request["params"] = processed_state.context.get("arguments", arguments)
        return request

    return request_middleware


def create_response_middleware(
    strukt_middleware: List[Middleware],
) -> Callable[..., Any]:
    """Create FastMCP response middleware that applies after_handle hooks."""

    def response_middleware(
        response: Dict[str, Any], request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """FastMCP response middleware."""
        # Extract tool information from request
        tool_name = request.get("method", "unknown")
        arguments = request.get("params", {})

        # Create InvocationState
        state = InvocationState(
            text=str(arguments),
            context={
                "tool_name": tool_name,
                "arguments": arguments,
                "source": "fastmcp_response",
            },
        )

        # Create HandlerResult from response
        result_content = response.get("result", {})
        handler_result = HandlerResult(
            response=result_content,
            status="success" if "error" not in response else "error",
        )

        # Apply after_handle middleware
        final_result = handler_result
        for middleware in strukt_middleware:
            try:
                final_result = middleware.after_handle(state, tool_name, final_result)
            except Exception as e:
                _log.warning(
                    f"Error in response middleware {middleware.__class__.__name__}: {e}"
                )

        # Update response with processed result
        if hasattr(final_result, "response"):
            response["result"] = final_result.response

        return response

    return response_middleware


def create_approval_middleware(
    strukt_middleware: List[Middleware],
) -> Optional[Callable[..., Any]]:
    """Create approval middleware if any Strukt middleware requires approval."""

    # Check if any middleware has approval logic
    approval_middleware = None
    for middleware in strukt_middleware:
        if hasattr(middleware, "should_run_background") and callable(
            getattr(middleware, "should_run_background")
        ):
            approval_middleware = middleware
            break

    if not approval_middleware:
        return None

    def approval_check(tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Check if tool execution requires approval."""
        state = InvocationState(
            text=str(arguments),
            context={
                "tool_name": tool_name,
                "arguments": arguments,
                "source": "fastmcp_approval",
            },
        )

        try:
            return approval_middleware.should_run_background(
                state, tool_name, [str(arguments)]
            )
        except Exception as e:
            _log.warning(f"Error in approval check: {e}")
            return False

    return approval_check


def create_logging_middleware(
    strukt_middleware: List[Middleware],
) -> Callable[..., Any]:
    """Create logging middleware that captures tool execution."""

    def logging_middleware(handler: Callable[..., Any]) -> Callable[..., Any]:
        """FastMCP logging middleware."""

        @functools.wraps(handler)
        async def logging_wrapper(*args, **kwargs) -> Any:
            """Wrapper that logs tool execution."""
            tool_name = kwargs.get("tool_name", "unknown")
            arguments = {k: v for k, v in kwargs.items() if k != "tool_name"}

            _log.info(
                f"Executing FastMCP tool '{tool_name}' with arguments: {list(arguments.keys())}"
            )

            try:
                result = (
                    await handler(*args, **kwargs)
                    if asyncio.iscoroutinefunction(handler)
                    else handler(*args, **kwargs)
                )
                _log.info(f"Tool '{tool_name}' executed successfully")
                return result
            except Exception as e:
                _log.error(f"Tool '{tool_name}' execution failed: {e}")
                raise

        return logging_wrapper

    return logging_middleware
