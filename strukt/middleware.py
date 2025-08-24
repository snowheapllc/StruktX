from __future__ import annotations

from abc import ABC
from typing import List, Tuple

from .types import HandlerResult, InvocationState, QueryClassification


class Middleware(ABC):
    """Extensible middleware with optional hooks.

    Subclasses can override any subset of hooks.
    """

    def before_classify(self, state: InvocationState) -> InvocationState:  # noqa: D401
        return state

    def after_classify(
        self,
        state: InvocationState,
        classification: QueryClassification,
    ) -> Tuple[InvocationState, QueryClassification]:  # noqa: D401
        return state, classification

    def before_handle(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> Tuple[InvocationState, List[str]]:  # noqa: D401
        return state, parts

    def after_handle(
        self,
        state: InvocationState,
        query_type: str,
        result: HandlerResult,
    ) -> HandlerResult:  # noqa: D401
        return result


def apply_before_classify(
    middleware: List[Middleware], state: InvocationState
) -> InvocationState:
    for m in middleware:
        state = m.before_classify(state)
    return state


def apply_after_classify(
    middleware: List[Middleware],
    state: InvocationState,
    classification: QueryClassification,
) -> Tuple[InvocationState, QueryClassification]:
    for m in middleware:
        state, classification = m.after_classify(state, classification)
    return state, classification


def apply_before_handle(
    middleware: List[Middleware],
    state: InvocationState,
    query_type: str,
    parts: List[str],
) -> Tuple[InvocationState, List[str]]:
    for m in middleware:
        state, parts = m.before_handle(state, query_type, parts)
    return state, parts


def apply_after_handle(
    middleware: List[Middleware],
    state: InvocationState,
    query_type: str,
    result: HandlerResult,
) -> HandlerResult:
    for m in middleware:
        result = m.after_handle(state, query_type, result)
    return result
