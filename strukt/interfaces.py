from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol, runtime_checkable

from .types import InvocationState, QueryClassification, HandlerResult


@runtime_checkable
class LLMClient(Protocol):
    def invoke(self, prompt: str, **kwargs: Any) -> Any: ...
    def structured(self, prompt: str, output_model: Any, **kwargs: Any) -> Any: ...


class Classifier(ABC):
    @abstractmethod
    def classify(self, state: InvocationState) -> QueryClassification:
        ...


class Handler(ABC):
    """A handler that receives the invocation state and the parts for its type."""

    @abstractmethod
    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        ...


class MemoryEngine(ABC):
    """Optional memory interface."""

    @abstractmethod
    async def process_webhook(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def cleanup(self, **kwargs: Any) -> Dict[str, Any]:
        ...



