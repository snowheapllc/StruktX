from __future__ import annotations

from abc import ABC
from typing import List, Optional, Tuple

from ..types import (
    HandlerResult,
    InvocationState,
    QueryClassification,
    BackgroundTaskInfo,
)


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

    def should_run_background(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> bool:  # noqa: D401
        """Determine if this handler should run in background."""
        return False

    def get_background_message(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> str:  # noqa: D401
        """Get the message to return immediately when running in background."""
        return "Task started in background. Use task tracking to monitor progress."

    def get_background_task_info(self, task_id: str) -> Optional[BackgroundTaskInfo]:  # noqa: D401
        """Get information about a specific background task."""
        return None

    def get_all_background_tasks(self) -> List[BackgroundTaskInfo]:  # noqa: D401
        """Get all background tasks."""
        return []

    def get_background_tasks_by_status(self, status: str) -> List[BackgroundTaskInfo]:  # noqa: D401
        """Get background tasks filtered by status."""
        return []


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


def apply_should_run_background(
    middleware: List[Middleware],
    state: InvocationState,
    query_type: str,
    parts: List[str],
) -> bool:
    """Check if any middleware wants to run this in background."""
    for m in middleware:
        if m.should_run_background(state, query_type, parts):
            return True
    return False


def apply_get_background_message(
    middleware: List[Middleware],
    state: InvocationState,
    query_type: str,
    parts: List[str],
) -> str:
    """Get background message from the first middleware that wants background execution."""
    for m in middleware:
        if m.should_run_background(state, query_type, parts):
            return m.get_background_message(state, query_type, parts)
    return "Task started in background. Use task tracking to monitor progress."


def apply_get_return_query_type(
    middleware: List[Middleware],
    state: InvocationState,
    query_type: str,
    parts: List[str],
) -> str:
    """Get return query type from the first middleware that wants background execution."""
    for m in middleware:
        if m.should_run_background(state, query_type, parts):
            if hasattr(m, "get_return_query_type"):
                return m.get_return_query_type(state, query_type, parts)
            # Fallback to original query type if middleware doesn't support custom return types
            return query_type
    return query_type


def apply_get_background_task_info(
    middleware: List[Middleware],
    task_id: str,
) -> Optional[BackgroundTaskInfo]:
    """Get background task information from middleware that supports it."""
    for m in middleware:
        if hasattr(m, "get_background_task_info"):
            result = m.get_background_task_info(task_id)
            if result is not None:
                return result
    return None


def apply_get_all_background_tasks(
    middleware: List[Middleware],
) -> List[BackgroundTaskInfo]:
    """Get all background tasks from middleware that supports it."""
    for m in middleware:
        if hasattr(m, "get_all_background_tasks"):
            return m.get_all_background_tasks()
    return []


def apply_get_background_tasks_by_status(
    middleware: List[Middleware],
    status: str,
) -> List[BackgroundTaskInfo]:
    """Get background tasks by status from middleware that supports it."""
    for m in middleware:
        if hasattr(m, "get_background_tasks_by_status"):
            return m.get_background_tasks_by_status(status)
    return []
