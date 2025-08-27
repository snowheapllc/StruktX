from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .interfaces import Classifier, Handler, MemoryEngine
from .middleware import (
    Middleware,
    apply_after_classify,
    apply_after_handle,
    apply_before_classify,
    apply_before_handle,
    apply_should_run_background,
    apply_get_background_message,
    apply_get_background_task_info,
    apply_get_all_background_tasks,
    apply_get_background_tasks_by_status,
)
from .types import (
    HandlerResult,
    InvocationState,
    QueryClassification,
    StruktQueryEnum,
    BackgroundTaskInfo,
)


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

        # Separate background and normal tasks
        normal_tasks = []

        for qtype, parts in grouped.items():
            handler = self._handlers.get(qtype) or fallback
            if handler is None:
                continue

            state, parts = apply_before_handle(self._middleware, state, qtype, parts)

            # Check if middleware wants to run this in background
            if apply_should_run_background(self._middleware, state, qtype, parts):
                # Get background message from middleware
                background_message = apply_get_background_message(
                    self._middleware, state, qtype, parts
                )

                # Create background task and return immediate response
                task_id = self._create_background_task(handler, state, qtype, parts)
                results.append(
                    HandlerResult(
                        response=background_message,
                        status=f"background_task_created:{task_id}",
                    )
                )
            else:
                # Add to normal tasks for parallel execution
                normal_tasks.append((qtype, parts, handler))

        # Execute normal tasks in parallel (regardless of background tasks)
        if normal_tasks:
            parallel_results = self._execute_handlers_parallel(state, normal_tasks)
            results.extend(parallel_results)

        return results

    def _execute_handlers_parallel(
        self, state: InvocationState, tasks: List[tuple[str, List[str], Handler]]
    ) -> List[HandlerResult]:
        """Execute multiple handlers in parallel."""
        results: List[HandlerResult] = []

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            future_to_task = {}
            for qtype, parts, handler in tasks:
                future = executor.submit(
                    self._execute_single_handler, state, qtype, parts, handler
                )
                future_to_task[future] = (qtype, parts, handler)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                qtype, parts, handler = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Handle any exceptions from parallel execution
                    results.append(
                        HandlerResult(
                            response=f"Error executing {qtype}: {str(e)}",
                            status="error",
                        )
                    )

        return results

    def _execute_single_handler(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
        handler: Handler,
    ) -> HandlerResult:
        """Execute a single handler and apply middleware."""
        try:
            result = handler.handle(state, parts)
            return apply_after_handle(self._middleware, state, query_type, result)
        except Exception as e:
            return HandlerResult(
                response=f"Error executing {query_type}: {str(e)}", status="error"
            )

    def _create_background_task(
        self,
        handler: Handler,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> str:
        """Create a background task for handler execution."""
        # Get the background task middleware if it exists
        background_middleware = self._get_background_task_middleware()
        if background_middleware:
            return background_middleware.create_background_task(
                handler, state, query_type, parts
            )
        else:
            # Fallback: return a simple task ID
            import uuid

            return str(uuid.uuid4())

    def _get_background_task_middleware(self):
        """Get the background task middleware if it exists."""
        for middleware in self._middleware:
            if hasattr(middleware, "create_background_task"):
                return middleware
        return None

    def get_background_task_info(self, task_id: str) -> Optional[BackgroundTaskInfo]:
        """Get information about a specific background task."""
        return apply_get_background_task_info(self._middleware, task_id)

    def get_all_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all background tasks."""
        return apply_get_all_background_tasks(self._middleware)

    def get_background_tasks_by_status(self, status: str) -> List[BackgroundTaskInfo]:
        """Get background tasks filtered by status."""
        return apply_get_background_tasks_by_status(self._middleware, status)

    def get_running_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all currently running background tasks."""
        return self.get_background_tasks_by_status("running")

    def get_completed_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all completed background tasks."""
        return self.get_background_tasks_by_status("completed")

    def get_failed_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all failed background tasks."""
        return self.get_background_tasks_by_status("failed")
