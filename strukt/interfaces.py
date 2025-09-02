from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, runtime_checkable

from .types import HandlerResult, InvocationState, QueryClassification


@runtime_checkable
class LLMClient(Protocol):
    def invoke(self, prompt: str, **kwargs: Any) -> Any: ...
    def structured(self, prompt: str, output_model: Any, **kwargs: Any) -> Any: ...


class Classifier(ABC):
    @abstractmethod
    def classify(self, state: InvocationState) -> QueryClassification: ...


class Handler(ABC):
    """A handler that receives the invocation state and the parts for its type."""

    @abstractmethod
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult: ...


class MemoryEngine(ABC):
    """Memory interface for adding, retrieving, and removing items."""

    @abstractmethod
    def add(self, text: str, metadata: Dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def get(self, query: str, top_k: int = 5) -> List[str]: ...

    @abstractmethod
    def get_scoped(
        self,
        query: str,
        *,
        user_id: str | None = None,
        unit_id: str | None = None,
        top_k: int = 5,
    ) -> List[str]:
        """Retrieve using engine-level metadata filters if supported.

        Implementations should fall back to unfiltered get() when filtering
        is not supported or when all scope values are None.
        """
        ...

    @abstractmethod
    def remove(self, text: str) -> int: ...

    @abstractmethod
    def cleanup(self, **kwargs: Any) -> Dict[str, Any]: ...


@runtime_checkable
class MCPExposable(Protocol):
    """Optional protocol for handlers that want to customize MCP exposure.

    Handlers may implement these attributes/methods to provide rich MCP tool
    metadata and a callable entrypoint.
    """

    # Human-friendly description for the MCP tool
    mcp_description: str  # type: ignore[assignment]
    # JSON Schema for input parameters
    mcp_parameters_schema: Dict[str, Any]  # type: ignore[assignment]
    # Optional human-readable title for display
    mcp_title: str | None  # type: ignore[assignment]
    # Optional output schema for structured results
    mcp_output_schema: Dict[str, Any] | None  # type: ignore[assignment]

    # The callable to execute when the tool is invoked by the MCP host
    def mcp_handle(self, **kwargs: Any) -> Any: ...
