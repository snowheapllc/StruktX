from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class APIKeyAuthConfig:
    header_name: str = "x-api-key"
    env_var: str = "STRUKTX_MCP_API_KEY"


class APIKeyAuthorizer:
    def __init__(self, cfg: APIKeyAuthConfig) -> None:
        self._cfg = cfg

    def is_authorized(self, headers: dict[str, str]) -> bool:
        provided = headers.get(self._cfg.header_name) or headers.get(
            self._cfg.header_name.upper()
        )
        if not provided:
            return False
        expected = os.environ.get(self._cfg.env_var)
        return bool(expected) and provided == expected
