from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from .types import HandlerResult, InvocationState, QueryClassification


@runtime_checkable
class LLMClient(Protocol):
    """LLM Client protocol with sync and async support."""

    def invoke(self, prompt: str, **kwargs: Any) -> Any: ...
    def structured(self, prompt: str, output_model: Any, **kwargs: Any) -> Any: ...

    # Async methods (optional for backward compatibility)
    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any: ...
    async def astructured(
        self, prompt: str, output_model: Any, **kwargs: Any
    ) -> Any: ...

    # Streaming methods (optional)
    async def astream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]: ...


class Classifier(ABC):
    """Classifier with sync and async support."""

    @abstractmethod
    def classify(self, state: InvocationState) -> QueryClassification: ...

    async def aclassify(self, state: InvocationState) -> QueryClassification:
        """Async classify with default sync fallback."""
        import asyncio

        return await asyncio.to_thread(self.classify, state)


class Handler(ABC):
    """A handler that receives the invocation state and the parts for its type.

    Handlers can implement either sync (handle) or async (ahandle) or both.
    If only one is implemented, the other will be auto-generated.
    """

    def handle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        """Sync handle - implement this OR ahandle."""
        # Check if ahandle is overridden (not the default implementation)
        import asyncio

        # If ahandle is overridden, run it synchronously
        if "ahandle" in self.__class__.__dict__:
            # Run the async version in the current event loop or create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, can't use run()
                    raise NotImplementedError(
                        f"{self.__class__.__name__} must implement handle() or use ainvoke() for async handlers"
                    )
                return loop.run_until_complete(self.ahandle(state, parts))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.ahandle(state, parts))

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either handle() or ahandle()"
        )

    async def ahandle(self, state: InvocationState, parts: list[str]) -> HandlerResult:
        """Async handle - implement this OR handle."""
        import asyncio

        # Check if handle is overridden (not the default implementation)
        if "handle" in self.__class__.__dict__:
            # Call sync version in a thread
            return await asyncio.to_thread(self.handle, state, parts)

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either handle() or ahandle()"
        )


class MemoryEngine(ABC):
    """Memory interface for adding, retrieving, and removing items."""

    @abstractmethod
    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None: ...

    @abstractmethod
    def get(self, query: str, top_k: int = 5) -> list[str]: ...

    @abstractmethod
    def get_scoped(
        self,
        query: str,
        *,
        user_id: str | None = None,
        unit_id: str | None = None,
        top_k: int = 5,
    ) -> list[str]:
        """Retrieve using engine-level metadata filters if supported.

        Implementations should fall back to unfiltered get() when filtering
        is not supported or when all scope values are None.
        """
        ...

    @abstractmethod
    def remove(self, text: str) -> int: ...

    @abstractmethod
    def cleanup(self, **kwargs: Any) -> dict[str, Any]: ...

    # Async methods with default sync fallback
    async def aadd(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Async add with default sync fallback."""
        import asyncio

        return await asyncio.to_thread(self.add, text, metadata)

    async def aget(self, query: str, top_k: int = 5) -> list[str]:
        """Async get with default sync fallback."""
        import asyncio

        return await asyncio.to_thread(self.get, query, top_k)

    async def aget_scoped(
        self,
        query: str,
        *,
        user_id: str | None = None,
        unit_id: str | None = None,
        top_k: int = 5,
    ) -> list[str]:
        """Async get_scoped with default sync fallback."""
        import asyncio

        return await asyncio.to_thread(
            self.get_scoped, query, user_id=user_id, unit_id=unit_id, top_k=top_k
        )


@runtime_checkable
class MCPExposable(Protocol):
    """Optional protocol for handlers that want to customize MCP exposure.

    Handlers may implement these attributes/methods to provide rich MCP tool
    metadata and a callable entrypoint.
    """

    # Human-friendly description for the MCP tool
    mcp_description: str  # type: ignore[assignment]
    # JSON Schema for input parameters
    mcp_parameters_schema: dict[str, Any]  # type: ignore[assignment]
    # Optional human-readable title for display
    mcp_title: str | None  # type: ignore[assignment]
    # Optional output schema for structured results
    mcp_output_schema: dict[str, Any] | None  # type: ignore[assignment]

    # The callable to execute when the tool is invoked by the MCP host
    def mcp_handle(self, **kwargs: Any) -> Any: ...
