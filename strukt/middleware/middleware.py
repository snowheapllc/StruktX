from __future__ import annotations

from abc import ABC

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
    ) -> tuple[InvocationState, QueryClassification]:  # noqa: D401
        return state, classification

    def before_handle(
        self,
        state: InvocationState,
        query_type: str,
        parts: list[str],
    ) -> tuple[InvocationState, list[str]]:  # noqa: D401
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
        parts: list[str],
    ) -> bool:  # noqa: D401
        """Determine if this handler should run in background."""
        return False

    def get_background_message(
        self,
        state: InvocationState,
        query_type: str,
        parts: list[str],
    ) -> str:  # noqa: D401
        """Get the message to return immediately when running in background."""
        return "Task started in background. Use task tracking to monitor progress."

    def get_background_task_info(self, task_id: str) -> BackgroundTaskInfo | None:  # noqa: D401
        """Get information about a specific background task."""
        return None

    def get_all_background_tasks(self) -> list[BackgroundTaskInfo]:  # noqa: D401
        """Get all background tasks."""
        return []

    def get_background_tasks_by_status(self, status: str) -> list[BackgroundTaskInfo]:  # noqa: D401
        """Get background tasks filtered by status."""
        return []


def apply_before_classify(
    middleware: list[Middleware], state: InvocationState
) -> InvocationState:
    for m in middleware:
        state = m.before_classify(state)
    return state


def apply_after_classify(
    middleware: list[Middleware],
    state: InvocationState,
    classification: QueryClassification,
) -> tuple[InvocationState, QueryClassification]:
    for m in middleware:
        state, classification = m.after_classify(state, classification)
    return state, classification


def apply_before_handle(
    middleware: list[Middleware],
    state: InvocationState,
    query_type: str,
    parts: list[str],
) -> tuple[InvocationState, list[str]]:
    for m in middleware:
        state, parts = m.before_handle(state, query_type, parts)
    return state, parts


def apply_after_handle(
    middleware: list[Middleware],
    state: InvocationState,
    query_type: str,
    result: HandlerResult,
) -> HandlerResult:
    for m in middleware:
        result = m.after_handle(state, query_type, result)
    return result


def apply_should_run_background(
    middleware: list[Middleware],
    state: InvocationState,
    query_type: str,
    parts: list[str],
) -> bool:
    """Check if any middleware wants to run this in background."""
    for m in middleware:
        if m.should_run_background(state, query_type, parts):
            return True
    return False


def apply_get_background_message(
    middleware: list[Middleware],
    state: InvocationState,
    query_type: str,
    parts: list[str],
) -> str:
    """Get background message from the first middleware that wants background execution."""
    for m in middleware:
        if m.should_run_background(state, query_type, parts):
            return m.get_background_message(state, query_type, parts)
    return "Task started in background. Use task tracking to monitor progress."


def apply_get_return_query_type(
    middleware: list[Middleware],
    state: InvocationState,
    query_type: str,
    parts: list[str],
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
    middleware: list[Middleware],
    task_id: str,
) -> BackgroundTaskInfo | None:
    """Get background task information from middleware that supports it."""
    for m in middleware:
        if hasattr(m, "get_background_task_info"):
            result = m.get_background_task_info(task_id)
            if result is not None:
                return result
    return None


def apply_get_all_background_tasks(
    middleware: list[Middleware],
) -> list[BackgroundTaskInfo]:
    """Get all background tasks from middleware that supports it."""
    for m in middleware:
        if hasattr(m, "get_all_background_tasks"):
            return m.get_all_background_tasks()
    return []


def apply_get_background_tasks_by_status(
    middleware: list[Middleware],
    status: str,
) -> list[BackgroundTaskInfo]:
    """Get background tasks by status from middleware that supports it."""
    for m in middleware:
        if hasattr(m, "get_background_tasks_by_status"):
            return m.get_background_tasks_by_status(status)
    return []
