"""Roomi extension middleware for background task management."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from strukt.middleware import Middleware
from strukt.types import HandlerResult, InvocationState, BackgroundTaskInfo
from strukt.logging import get_logger
from strukt.tracing import strukt_trace, unified_trace_context


class TaskStatus(Enum):
    """Status of background tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """Represents a background task."""

    task_id: str
    handler_name: str
    handler_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[HandlerResult] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    future: Optional[Future] = None


class BackgroundTaskMiddleware(Middleware):
    """Middleware for managing background tasks.
    This middleware intercepts handler calls and can run them in background threads,
    returning immediate responses while tracking task progress.
    """

    def __init__(
        self,
        max_workers: int = 4,
        default_message: str = "Task started in background. Use task tracking to monitor progress.",
        enable_background_for: Optional[Set[str]] = None,
        disable_background_for: Optional[Set[str]] = None,
        custom_messages: Optional[Dict[str, str]] = None,
        action_based_background: Optional[Dict[str, Set[str]]] = None,
        return_query_types: Optional[Dict[str, str]] = None,
    ):
        self._log = get_logger("background_task_manager")
        self.max_workers = max_workers
        self.default_message = default_message
        self.enable_background_for = enable_background_for or set()
        self.disable_background_for = disable_background_for or set()
        self.custom_messages = custom_messages or {}
        self.action_based_background = action_based_background or {}
        self.return_query_types = return_query_types or {}

        # Task management
        self._tasks: Dict[str, BackgroundTask] = {}
        # Prefer Weave's ThreadPoolExecutor to propagate trace context across threads
        try:
            import weave  # type: ignore

            self._executor = weave.ThreadPoolExecutor(max_workers=max_workers)
        except Exception:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

        # Background thread for cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def should_run_background(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> bool:
        """Determine if this handler should run in background."""
        # Check if background execution is requested via context
        run_background = state.context.get("run_background", False)
        if run_background:
            return True

        # Auto-detect if this should run in background based on query type and content
        return self._should_run_background(query_type, parts, state)

    def get_background_message(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> str:
        """Get the message to return immediately when running in background."""
        background_message = state.context.get("background_message", None)
        if background_message:
            return background_message

        return self._get_background_message(query_type, parts)

    def get_return_query_type(
        self,
        state: InvocationState,
        query_type: str,
        parts: List[str],
    ) -> str:
        """Get the query type to return when running in background."""
        # Check if handler specified a return query type in context
        # Support both dict format (handler_name: query_type) and single value for backward compatibility
        return_query_types = state.context.get("return_query_types", {})
        if isinstance(return_query_types, dict) and query_type in return_query_types:
            handler_return_query_type = return_query_types[query_type]
            self._log.debug(
                f"[BACKGROUND TASK] Handler specified return query type for '{query_type}': {handler_return_query_type}"
            )
            return handler_return_query_type

        # Fallback to single return_query_type for backward compatibility
        handler_return_query_type = state.context.get("return_query_type", None)
        if handler_return_query_type:
            self._log.debug(
                f"[BACKGROUND TASK] Handler specified return query type: {handler_return_query_type}"
            )
            return handler_return_query_type

        # Check if there's a configured return query type for this handler
        if query_type in self.return_query_types:
            configured_return_type = self.return_query_types[query_type]
            self._log.debug(
                f"[BACKGROUND TASK] Using configured return query type for '{query_type}': {configured_return_type}"
            )
            return configured_return_type

        # Fallback to the original query type
        self._log.debug(
            f"[BACKGROUND TASK] Using original query type as return type: {query_type}"
        )
        return query_type

    def _should_run_background(
        self, query_type: str, parts: List[str], state: InvocationState
    ) -> bool:
        """Determine if a task should run in background based on type and content."""
        self._log.debug(
            f"[BACKGROUND TASK] Checking if '{query_type}' should run in background"
        )

        # Check if this query type is explicitly enabled for background execution
        if self.enable_background_for and query_type in self.enable_background_for:
            self._log.debug(
                f"[BACKGROUND TASK] Query type '{query_type}' is explicitly enabled for background"
            )
            return True

        # Check if this query type is explicitly disabled
        if self.disable_background_for and query_type in self.disable_background_for:
            self._log.debug(
                f"[BACKGROUND TASK] Query type '{query_type}' is explicitly disabled for background"
            )
            return False

        # Check action-based background execution
        if query_type in self.action_based_background:
            self._log.debug(
                f"[BACKGROUND TASK] Query type '{query_type}' has action-based background config: {self.action_based_background[query_type]}"
            )
            action = self._extract_action_from_parts(parts, state, query_type)
            if action and action in self.action_based_background[query_type]:
                self._log.debug(
                    f"[BACKGROUND TASK] Action '{action}' matches action-based background config"
                )
                return True
            else:
                self._log.debug(
                    f"[BACKGROUND TASK] Action '{action}' does not match action-based background config"
                )

        # Default behavior for specific query types
        if query_type == "device_control":
            self._log.debug(
                f"[BACKGROUND TASK] Query type '{query_type}' has default background behavior"
            )
            return True

        self._log.debug(
            f"[BACKGROUND TASK] Query type '{query_type}' will run in foreground"
        )
        return False

    def _extract_action_from_parts(
        self, parts: List[str], state: InvocationState, query_type: str
    ) -> Optional[str]:
        """Extract action from handler context."""
        # Get handler intents dict from context (handlers set their intents here)
        handler_intents = state.context.get("handler_intents", {})
        if not handler_intents:
            self._log.debug("[BACKGROUND TASK] No handler intents found in context")
            return None

        # Get the action for the current query_type (handler name)
        action = handler_intents.get(query_type)
        if action:
            self._log.debug(
                f"[BACKGROUND TASK] Action extracted from context for handler '{query_type}': {action}"
            )
            return action

        self._log.debug(
            f"[BACKGROUND TASK] No action found for handler '{query_type}' in context"
        )
        return None

    def _get_background_message(self, query_type: str, parts: List[str]) -> str:
        """Get appropriate background message based on query type."""
        # Check for custom message first
        if query_type in self.custom_messages:
            return self.custom_messages[query_type]

        # Fallback to default messages
        if query_type == "device_control":
            return "I'm processing your device control request in the background. This may take a moment."
        elif query_type == "maintenance_or_helpdesk":
            return "I'm creating your helpdesk ticket in the background. You'll receive a confirmation shortly."
        else:
            return self.default_message

    def create_background_task(
        self, handler, state: InvocationState, query_type: str, parts: List[str]
    ) -> str:
        """Create a new background task."""
        task_id = str(uuid.uuid4())

        # Capture current trace context to pass to background thread
        current_thread_id = None
        try:
            ctx = getattr(state, "context", {}) or {}
            current_thread_id = ctx.get("thread_id") or ctx.get("session_id")
        except Exception:
            current_thread_id = None

        task = BackgroundTask(
            task_id=task_id,
            handler_name=query_type,
            handler_id=query_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata={
                "query_type": query_type,
                "parts": parts,
                "original_state": state,
                "parent_thread_id": current_thread_id,  # Store parent thread context
            },
        )

        with self._lock:
            self._tasks[task_id] = task

        # Submit task to thread pool
        future = self._executor.submit(
            self._execute_background_task,
            task_id,
            handler,
            query_type,
            parts,
            state,
            current_thread_id,  # Pass parent thread context
        )
        task.future = future

        self._log.info(
            f"[BACKGROUND TASK] Created task {task_id} for handler '{query_type}'"
        )
        return task_id

    @strukt_trace(
        name="StruktX.BackgroundTask.execute",
        call_display_name="BackgroundTask.execute",
    )
    def _execute_background_task(
        self,
        task_id: str,
        handler,
        query_type: str,
        parts: List[str],
        state: InvocationState,
        parent_thread_id: str = None,
    ) -> dict:
        """Execute a task in the background."""
        with self._lock:
            if task_id not in self._tasks:
                return {"error": "Task not found", "task_id": task_id}
            task = self._tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

        try:
            self._log.info(
                f"[BACKGROUND TASK] Executing task {task_id} for '{query_type}'"
            )

            # Update progress to indicate start
            with self._lock:
                if task_id in self._tasks:
                    self._tasks[task_id].progress = 0.1

            # Use the parent thread context to ensure nesting under main trace
            thread_id = parent_thread_id
            if not thread_id:
                try:
                    ctx = getattr(state, "context", {}) or {}
                    thread_id = ctx.get("thread_id") or ctx.get("session_id")
                except Exception:
                    thread_id = None

            # Execute the actual handler within the parent trace context
            # This ensures the background task appears nested under the main Engine.run
            with unified_trace_context(thread_id, f"BackgroundTask.{query_type}"):
                result = handler.handle(state, parts)

            # Update progress to indicate completion
            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.progress = 1.0
                    task.result = result

            self._log.info(
                f"[BACKGROUND TASK] Task {task_id} completed successfully for '{query_type}'"
            )

            # Return the result for tracing capture
            result_dict = (
                result.__dict__ if hasattr(result, "__dict__") else str(result)
            )
            return {
                "task_id": task_id,
                "query_type": query_type,
                "status": "completed",
                "result": result_dict,
                "parts": parts,
                "user_id": state.context.get("user_id")
                if hasattr(state, "context")
                else None,
            }

        except Exception as e:
            self._log.error(
                f"[BACKGROUND TASK] Task {task_id} failed for '{query_type}': {e}"
            )

            with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    task.error = str(e)

            # Return the error for tracing capture
            return {
                "task_id": task_id,
                "query_type": query_type,
                "status": "failed",
                "error": str(e),
                "parts": parts,
                "user_id": state.context.get("user_id")
                if hasattr(state, "context")
                else None,
            }

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Get a specific task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[BackgroundTask]:
        """Get all tasks."""
        with self._lock:
            return list(self._tasks.values())

    def get_tasks_by_status(self, status: TaskStatus) -> List[BackgroundTask]:
        """Get tasks filtered by status."""
        with self._lock:
            return [task for task in self._tasks.values() if task.status == status]

    def get_background_task_info(self, task_id: str) -> Optional[BackgroundTaskInfo]:
        """Get information about a specific background task."""
        task = self.get_task(task_id)
        if not task:
            return None

        return {
            "task_id": task.task_id,
            "handler_name": task.handler_name,
            "handler_id": task.handler_id,
            "status": task.status.value,
            "progress": task.progress,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": (
                task.completed_at.isoformat() if task.completed_at else None
            ),
            "result": asdict(task.result) if task.result else None,
            "error": task.error,
            "metadata": task.metadata,
        }

    def get_all_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all background tasks."""
        tasks = self.get_all_tasks()
        return [
            {
                "task_id": task.task_id,
                "handler_name": task.handler_name,
                "handler_id": task.handler_id,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "result": asdict(task.result) if task.result else None,
                "error": task.error,
            }
            for task in tasks
        ]

    def get_background_tasks_by_status(self, status: str) -> List[BackgroundTaskInfo]:
        """Get background tasks filtered by status."""
        try:
            task_status = TaskStatus(status)
        except ValueError:
            return []

        tasks = self.get_tasks_by_status(task_status)
        return [
            {
                "task_id": task.task_id,
                "handler_name": task.handler_name,
                "handler_id": task.handler_id,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": (
                    task.completed_at.isoformat() if task.completed_at else None
                ),
                "result": asdict(task.result) if task.result else None,
                "error": task.error,
            }
            for task in tasks
        ]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            if task.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                return False

            if task.future and not task.future.done():
                task.future.cancel()

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

            self._log.info(f"⏹[BACKGROUND TASK] Cancelled task {task_id}")
            return True

    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up completed tasks older than specified age."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0

        with self._lock:
            tasks_to_remove = []
            for task_id, task in self._tasks.items():
                if (
                    task.status
                    in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                    and task.completed_at
                    and task.completed_at.timestamp() < cutoff_time
                ):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                cleaned_count += 1

        if cleaned_count > 0:
            self._log.info(f"[BACKGROUND TASK] Cleaned up {cleaned_count} old tasks")

        return cleaned_count

    def _cleanup_loop(self) -> None:
        """Background thread for periodic cleanup."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self.cleanup_completed_tasks()
            except Exception as e:
                self._log.error(f"[BACKGROUND TASK] Error in cleanup loop: {e}")

    def shutdown(self) -> None:
        """Shutdown the middleware and cancel all tasks."""
        self._log.info("[BACKGROUND TASK] Shutting down background task middleware")

        # Cancel all running tasks
        with self._lock:
            for task in self._tasks.values():
                if task.future and not task.future.done():
                    task.future.cancel()

        # Shutdown executor
        self._executor.shutdown(wait=True)

        self._log.info("[BACKGROUND TASK] Background task middleware shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if hasattr(self, "_log"):
                self.shutdown()
        except Exception as e:
            if hasattr(self, "_log"):
                self._log.error(
                    f"[BACKGROUND TASK] Error in background task middleware shutdown: {e}"
                )
