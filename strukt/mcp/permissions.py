from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from ..interfaces import MemoryEngine


class ConsentPolicy:
    ALWAYS_ASK = "always-ask"
    ASK_ONCE = "ask-once"
    ALWAYS_ALLOW = "always-allow"
    NEVER_ALLOW = "never-allow"


@dataclass
class ConsentDecision:
    user_id: str
    tool_name: str
    decision: str  # values from ConsentPolicy


class ConsentStore:
    """Consent storage backed by MemoryEngine if available, else in-memory.

    Keys: f"consent:{user_id}:{tool_name}" -> decision
    """

    def __init__(self, memory: MemoryEngine | None = None) -> None:
        self._memory = memory
        self._cache: Dict[str, str] = {}

    def _key(self, user_id: str, tool_name: str) -> str:
        return f"consent:{user_id}:{tool_name}"

    def set(self, decision: ConsentDecision) -> None:
        key = self._key(decision.user_id, decision.tool_name)
        self._cache[key] = decision.decision
        # Persist a compact representation when memory is provided
        if self._memory is not None:
            self._memory.add(key, {"decision": decision.decision})

    def get(self, user_id: str, tool_name: str) -> Optional[str]:
        key = self._key(user_id, tool_name)
        if key in self._cache:
            return self._cache[key]
        # Try to recover from memory store
        if self._memory is not None:
            results = self._memory.get_scoped(
                key, user_id=user_id, unit_id=None, top_k=1
            )
            if results:
                # best-effort parse; store in cache
                self._cache[key] = (
                    ConsentPolicy.ALWAYS_ALLOW
                    if "always-allow" in results[0]
                    else (
                        ConsentPolicy.NEVER_ALLOW
                        if "never-allow" in results[0]
                        else (
                            ConsentPolicy.ASK_ONCE
                            if "ask-once" in results[0]
                            else (
                                ConsentPolicy.ALWAYS_ASK
                                if "always-ask" in results[0]
                                else ""
                            )
                        )
                    )
                )
                return self._cache.get(key)
        return None
