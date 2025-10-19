"""
Authentication components for FastMCP v2 integration.

This module provides authentication utilities for FastMCP v2,
replacing the old MCP v1 authentication system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from ..logging import get_logger

_log = get_logger("mcp_v2.auth")


@dataclass
class APIKeyAuthConfig:
    """Configuration for API key authentication."""

    enabled: bool = False
    header_name: str = "x-api-key"
    env_var: str = "STRUKTX_MCP_API_KEY"


class APIKeyAuthorizer:
    """API key authorizer for MCP v2 authentication."""

    def __init__(self, cfg: APIKeyAuthConfig) -> None:
        """Initialize the API key authorizer.

        Args:
            cfg: API key authentication configuration
        """
        self._cfg = cfg
        _log.debug(
            f"Initialized API key authorizer with header '{cfg.header_name}' and env var '{cfg.env_var}'"
        )

    def is_authorized(self, headers: Dict[str, str]) -> bool:
        """Check if the request is authorized based on API key.

        Args:
            headers: Request headers dictionary

        Returns:
            True if authorized, False otherwise
        """
        if not self._cfg.enabled:
            _log.debug("API key authentication disabled")
            return True

        # Get the provided API key from headers
        provided_key = headers.get(self._cfg.header_name) or headers.get(
            self._cfg.header_name.upper()
        )

        if not provided_key:
            _log.warning(f"Missing API key in header '{self._cfg.header_name}'")
            return False

        # Get the expected API key from environment
        expected_key = os.environ.get(self._cfg.env_var)

        if not expected_key:
            _log.warning(f"API key environment variable '{self._cfg.env_var}' not set")
            return False

        # Compare keys
        is_valid = provided_key == expected_key

        if not is_valid:
            _log.warning("Invalid API key provided")
        else:
            _log.debug("API key authentication successful")

        return is_valid

    def get_expected_key(self) -> Optional[str]:
        """Get the expected API key from environment.

        Returns:
            The expected API key or None if not set
        """
        return os.environ.get(self._cfg.env_var)

    def is_configured(self) -> bool:
        """Check if authentication is properly configured.

        Returns:
            True if authentication is enabled and API key is set
        """
        if not self._cfg.enabled:
            return True  # Not required if disabled

        return bool(self.get_expected_key())
