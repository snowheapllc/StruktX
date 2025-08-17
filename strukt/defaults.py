from __future__ import annotations

from typing import Any, List, Type

from pydantic import BaseModel

from .interfaces import LLMClient, Classifier, Handler
from .types import InvocationState, QueryClassification, HandlerResult


class SimpleLLMClient(LLMClient):
    """A minimal LLM client placeholder. Users should supply their own.

    For development, this client just echoes prompts.
    """

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        return type("Resp", (), {"content": prompt})

    def structured(self, prompt: str, output_model: Type[BaseModel], **kwargs: Any) -> Any:
        # Create a trivial instance with defaults if possible
        try:
            return output_model()
        except Exception:
            # Try pydantic v2 then v1 fallback
            try:
                return output_model.model_construct()  # type: ignore[attr-defined]
            except Exception:
                return output_model.construct()  # type: ignore[attr-defined]


class SimpleClassifier(Classifier):
    def classify(self, state: InvocationState) -> QueryClassification:
        # Default: route everything to 'general'
        return QueryClassification(query_types=["general"], confidences=[1.0], parts=[state.text])


class GeneralHandler(Handler):
    def __init__(self, prompt_template: str | None = None) -> None:
        self._prompt = prompt_template or "{text}"

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        text = parts[0] if parts else state.text
        response = self._prompt.format(text=text, context=state.context)
        return HandlerResult(response=str(response), status="GENERAL")


