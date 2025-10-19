"""
Authentication adapter for FastMCP v2 integration.

This module provides utilities to adapt Strukt's authentication system
to FastMCP's middleware patterns.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict
import functools
import asyncio

from .auth import APIKeyAuthConfig, APIKeyAuthorizer
from ..logging import get_logger

_log = get_logger("mcp_v2.auth_adapter")


def create_auth_middleware(auth_config: APIKeyAuthConfig) -> Callable[..., Any]:
    """Create FastMCP-compatible auth middleware from Strukt config.

    Args:
        auth_config: Strukt API key authentication configuration

    Returns:
        FastMCP-compatible middleware function
    """
    authorizer = APIKeyAuthorizer(auth_config)

    def auth_middleware(handler: Callable[..., Any]) -> Callable[..., Any]:
        """FastMCP auth middleware that checks API keys."""

        @functools.wraps(handler)
        async def auth_wrapper(*args, **kwargs) -> Any:
            """Wrapper that enforces authentication."""

            # Extract headers from FastMCP context
            # FastMCP may pass headers in different ways depending on transport
            headers = _extract_headers_from_context(kwargs)

            # Check authorization
            if not authorizer.is_authorized(headers):
                _log.warning("Unauthorized MCP request - missing or invalid API key")
                raise PermissionError("API key is required for all MCP requests")

            _log.debug("Authorized MCP request")

            # Execute the original handler
            return (
                await handler(*args, **kwargs)
                if asyncio.iscoroutinefunction(handler)
                else handler(*args, **kwargs)
            )

        return auth_wrapper

    _log.info(
        f"Created FastMCP auth middleware with header '{auth_config.header_name}'"
    )
    return auth_middleware


def _extract_headers_from_context(kwargs: Dict[str, Any]) -> Dict[str, str]:
    """Extract headers from FastMCP context.

    FastMCP may pass headers in different ways depending on the transport:
    - For HTTP/SSE: headers might be in request context
    - For stdio: headers might be in metadata or context
    """
    headers = {}

    # Try to extract from various possible locations
    if "headers" in kwargs:
        headers.update(kwargs["headers"])

    if "request" in kwargs and isinstance(kwargs["request"], dict):
        request = kwargs["request"]
        if "headers" in request:
            headers.update(request["headers"])

    if "context" in kwargs and isinstance(kwargs["context"], dict):
        context = kwargs["context"]
        if "headers" in context:
            headers.update(context["headers"])

    # For stdio transport, headers might be in metadata
    if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
        metadata = kwargs["metadata"]
        if "headers" in metadata:
            headers.update(metadata["headers"])

    return headers


def create_request_auth_middleware(auth_config: APIKeyAuthConfig) -> Callable[..., Any]:
    """Create FastMCP request-level auth middleware."""
    authorizer = APIKeyAuthorizer(auth_config)

    def request_auth_middleware(request: Dict[str, Any]) -> Dict[str, Any]:
        """FastMCP request auth middleware."""

        # Extract headers from request
        headers = request.get("headers", {})

        # Check authorization
        if not authorizer.is_authorized(headers):
            _log.warning("Unauthorized MCP request - missing or invalid API key")
            # Return error response
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32001, "message": "Unauthorized - API key required"},
            }

        _log.debug("Authorized MCP request")
        return request

    return request_auth_middleware


def create_environment_auth_check(auth_config: APIKeyAuthConfig) -> bool:
    """Check if authentication is properly configured in environment.

    Args:
        auth_config: Authentication configuration

    Returns:
        True if auth is properly configured, False otherwise
    """
    api_key = os.environ.get(auth_config.env_var)
    if not api_key:
        _log.warning(
            f"API key not found in environment variable '{auth_config.env_var}'"
        )
        return False

    _log.debug(f"API key found in environment variable '{auth_config.env_var}'")
    return True


def create_auth_error_handler(auth_config: APIKeyAuthConfig) -> Callable[..., Any]:
    """Create a standardized auth error handler for FastMCP."""

    def auth_error_handler(error: Exception) -> Dict[str, Any]:
        """Handle authentication errors consistently."""
        if isinstance(error, PermissionError):
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32001,
                    "message": "Unauthorized - API key required",
                    "data": {
                        "header_name": auth_config.header_name,
                        "env_var": auth_config.env_var,
                    },
                },
            }

        # Re-raise non-auth errors
        raise error

    return auth_error_handler


def create_optional_auth_middleware(
    auth_config: APIKeyAuthConfig,
) -> Callable[..., Any]:
    """Create optional auth middleware that warns but doesn't block if auth is not configured."""

    def optional_auth_middleware(handler: Callable[..., Any]) -> Callable[..., Any]:
        """FastMCP optional auth middleware."""

        @functools.wraps(handler)
        async def optional_auth_wrapper(*args, **kwargs) -> Any:
            """Wrapper that optionally enforces authentication."""

            # Check if auth is configured
            if not create_environment_auth_check(auth_config):
                _log.warning("Authentication not configured - allowing request")
                return (
                    await handler(*args, **kwargs)
                    if asyncio.iscoroutinefunction(handler)
                    else handler(*args, **kwargs)
                )

            # Auth is configured, enforce it
            headers = _extract_headers_from_context(kwargs)
            authorizer = APIKeyAuthorizer(auth_config)

            if not authorizer.is_authorized(headers):
                _log.warning("Unauthorized MCP request - missing or invalid API key")
                raise PermissionError("API key is required for all MCP requests")

            _log.debug("Authorized MCP request")
            return (
                await handler(*args, **kwargs)
                if asyncio.iscoroutinefunction(handler)
                else handler(*args, **kwargs)
            )

        return optional_auth_wrapper

    return optional_auth_middleware
