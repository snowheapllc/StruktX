from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .interfaces import Classifier, Handler, MemoryEngine
from .middleware import (
    Middleware,
    apply_after_classify,
    apply_after_handle,
    apply_before_classify,
    apply_before_handle,
)
from .types import HandlerResult, InvocationState, QueryClassification, StruktQueryEnum


class Engine:
    def __init__(
        self,
        *,
        classifier: Classifier,
        handlers: Dict[str, Handler],
        default_route: str | None = None,
        memory: MemoryEngine | None = None,
        middleware: list[Middleware] | None = None,
    ) -> None:
        self._classifier = classifier
        self._handlers = handlers
        self._default_route = default_route
        self._memory = memory
        self._middleware = list(middleware or [])

    def run(self, state: InvocationState) -> List[HandlerResult]:
        state, _ = self._classify(state)
        fallback = self._maybe_fallback_handler()
        if self._should_fallback(state):
            return [fallback.handle(state, [state.text])] if fallback else []

        grouped = self._group_parts_by_type(state)
        return self._execute_grouped_handlers(state, grouped, fallback)

    def _classify(
        self, state: InvocationState
    ) -> tuple[InvocationState, QueryClassification]:
        state = apply_before_classify(self._middleware, state)
        classification: QueryClassification = self._classifier.classify(state)
        state, classification = apply_after_classify(
            self._middleware, state, classification
        )
        state.query_types = list(classification.query_types)
        state.confidences = list(classification.confidences)
        state.parts = list(classification.parts)
        return state, classification

    def _maybe_fallback_handler(self) -> Handler | None:
        return self._handlers.get(self._default_route or StruktQueryEnum.GENERAL)

    def _should_fallback(self, state: InvocationState) -> bool:
        return not state.query_types or not state.parts

    def _group_parts_by_type(self, state: InvocationState) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for idx, qtype in enumerate(state.query_types):
            part = state.parts[idx] if idx < len(state.parts) else state.text
            grouped[qtype].append(part)
        return grouped

    def _execute_grouped_handlers(
        self,
        state: InvocationState,
        grouped: Dict[str, List[str]],
        fallback: Handler | None,
    ) -> List[HandlerResult]:
        results: List[HandlerResult] = []
        for qtype, parts in grouped.items():
            handler = self._handlers.get(qtype) or fallback
            if handler is None:
                continue
            state, parts = apply_before_handle(self._middleware, state, qtype, parts)
            result = handler.handle(state, parts)
            results.append(apply_after_handle(self._middleware, state, qtype, result))
        return results
