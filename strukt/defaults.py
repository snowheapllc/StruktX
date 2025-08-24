from __future__ import annotations

from typing import Any, List, Type

from pydantic import BaseModel

from .interfaces import Classifier, Handler, LLMClient, MemoryEngine
from .prompts import render_prompt_with_safe_braces
from .types import HandlerResult, InvocationState, QueryClassification, StruktQueryEnum


class SimpleLLMClient(LLMClient):
    """A minimal LLM client placeholder. Users should supply their own.

    For development, this client just echoes prompts.
    """

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        # No formatting here, this client is for dev only and echoes back
        return type("Resp", (), {"content": prompt})

    def structured(
        self, prompt: str, output_model: Type[BaseModel], **kwargs: Any
    ) -> Any:
        # Try to return an empty-but-valid object for common fields to avoid raising
        try:
            return output_model.model_validate(
                {
                    "query_types": [],
                    "confidences": [],
                    "parts": [],
                }
            )
        except Exception:
            # Best-effort pydantic v2 -> v1 fallback paths
            try:
                return output_model.model_construct(
                    query_types=[], confidences=[], parts=[]
                )  # type: ignore[attr-defined]
            except Exception:
                try:
                    return output_model.construct(
                        query_types=[], confidences=[], parts=[]
                    )  # type: ignore[attr-defined]
                except Exception:
                    return output_model()  # may still raise if strict


class MemoryAugmentedLLMClient(LLMClient):
    """Decorator that injects retrieved memory context into prompts."""

    def __init__(
        self, base: LLMClient, memory: MemoryEngine, *, top_k: int = 5
    ) -> None:
        self._base = base
        self._memory = memory
        self._top_k = top_k

    def _augment(
        self,
        prompt: str,
        *,
        context: dict | None = None,
        query_hint: str | None = None,
        source: str | None = None,
    ) -> str:
        try:
            # Prefer sync retrieval via new API; fallback to legacy method names if present
            docs: List[str] = []
            user_id = None
            unit_id = None
            if context:
                try:
                    user_id = (
                        str(context.get("user_id"))
                        if context.get("user_id") is not None
                        else None
                    )
                    unit_id = (
                        str(context.get("unit_id"))
                        if context.get("unit_id") is not None
                        else None
                    )
                except Exception:
                    user_id = None
                    unit_id = None
            # Try using store for scope-aware listing if present
            try:
                store = getattr(self._memory, "store", None)
            except Exception:
                store = None
            if store is not None and (user_id or unit_id):
                try:
                    docs = store.list_engine_memories_for_scope(
                        user_id=user_id, unit_id=unit_id, limit=self._top_k
                    )  # type: ignore[attr-defined]
                except Exception:
                    docs = []
            if not docs:
                retrieval_query = query_hint or prompt
                if hasattr(self._memory, "get_scoped") and callable(
                    self._memory.get_scoped
                ):
                    docs = self._memory.get_scoped(
                        retrieval_query,
                        user_id=user_id,
                        unit_id=unit_id,
                        top_k=self._top_k,
                    )  # type: ignore[call-arg]
                elif hasattr(self._memory, "get") and callable(self._memory.get):
                    docs = self._memory.get(retrieval_query, self._top_k)  # type: ignore[arg-type]
                elif hasattr(self._memory, "retrieve") and callable(
                    self._memory.retrieve
                ):
                    docs = self._memory.retrieve(
                        retrieval_query, self._top_k
                    )  # type: ignore[call-arg]
        except Exception:
            docs = []
        if not docs:
            return prompt
        mem_block = "\n".join(f"- {d}" for d in docs)
        # Log memory injection using our structured logger with source context
        try:
            from .logging import get_logger

            _log = get_logger("memory")
            src = source or "unknown"
            _log.memory_injection(src, len(docs))
        except Exception:
            pass
        return f"Relevant memory:\n{mem_block}\n\n{prompt}"

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        ctx = kwargs.get("context") if isinstance(kwargs, dict) else None
        qh = kwargs.get("query_hint") if isinstance(kwargs, dict) else None
        src = None
        if isinstance(kwargs, dict):
            src = kwargs.get("augment_source") or kwargs.get("caller")
        augmented = self._augment(prompt, context=ctx, query_hint=qh, source=src)
        # Remove augmentation-only kwargs before delegating
        clean_kwargs = dict(kwargs)
        if "context" in clean_kwargs:
            clean_kwargs.pop("context", None)
        if "query_hint" in clean_kwargs:
            clean_kwargs.pop("query_hint", None)
        if "augment_source" in clean_kwargs:
            clean_kwargs.pop("augment_source", None)
        if "caller" in clean_kwargs:
            clean_kwargs.pop("caller", None)
        return self._base.invoke(augmented, **clean_kwargs)

    def structured(self, prompt: str, output_model: Any, **kwargs: Any) -> Any:
        ctx = kwargs.get("context") if isinstance(kwargs, dict) else None
        qh = kwargs.get("query_hint") if isinstance(kwargs, dict) else None
        src = None
        if isinstance(kwargs, dict):
            src = kwargs.get("augment_source") or kwargs.get("caller")
        augmented = self._augment(prompt, context=ctx, query_hint=qh, source=src)
        clean_kwargs = dict(kwargs)
        if "context" in clean_kwargs:
            clean_kwargs.pop("context", None)
        if "query_hint" in clean_kwargs:
            clean_kwargs.pop("query_hint", None)
        if "augment_source" in clean_kwargs:
            clean_kwargs.pop("augment_source", None)
        if "caller" in clean_kwargs:
            clean_kwargs.pop("caller", None)
        return self._base.structured(augmented, output_model, **clean_kwargs)


class SimpleClassifier(Classifier):
    def classify(self, state: InvocationState) -> QueryClassification:
        # Default: route everything to 'general'
        return QueryClassification(
            query_types=[StruktQueryEnum.GENERAL], confidences=[1.0], parts=[state.text]
        )


class GeneralHandler(Handler):
    """General handler that consults the configured LLM for a friendly reply.

    If the app enabled memory augmentation, the LLM is already wrapped to include
    relevant context. On failure, we return the user text as-is.
    """

    def __init__(self, llm: LLMClient, prompt_template: str | None = None) -> None:
        self._llm = llm
        self._prompt = prompt_template

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        # Preserve the user's full question for general responses
        text = state.text
        try:
            if self._prompt:
                # Safely format only known variables, preserving any JSON braces in templates
                prompt = render_prompt_with_safe_braces(
                    self._prompt, {"text": text, "context": state.context}
                )
            else:
                prompt = (
                    "You are a helpful assistant. Provide a concise, friendly, and personalized answer.\n"
                    "Incorporate any 'Relevant memory' context provided above to avoid asking for information we already know.\n"
                    "Prefer to use known preferences (e.g., food, location) to answer directly.\n\n"
                    f"User: {text}\n"
                    "Assistant:"
                )
            resp = self._llm.invoke(
                prompt, context=state.context, query_hint=state.text
            )
            content = getattr(resp, "content", None)
            response = str(content) if content is not None else str(resp)
            if not response:
                response = text
            return HandlerResult(response=response, status=StruktQueryEnum.GENERAL)
        except Exception:
            return HandlerResult(response=text, status=StruktQueryEnum.GENERAL)
