from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, cast
import contextlib

from .config import OpenTelemetryConfig, TracingConfig

_T = TypeVar("_T")
F = TypeVar("F", bound=Callable[..., Any])

# Global state for unified tracing
_trace_enabled = True
_active_traces = {}  # Track active traces to wait for background tasks
_trace_locks = {}  # Locks per trace to ensure completion


def generate_trace_name(context: dict = None, prefix: str = None) -> str:
    """Generate a trace name in format: [prefix-]userID-unitID-UUID-timestamp

    Args:
        context: Context dict containing user_id, unit_id, thread_id/session_id
        prefix: Optional prefix to prepend to the trace name

    Returns:
        Formatted trace name string
    """
    if not context:
        context = {}

    user_id = context.get("user_id", "unknown")
    unit_id = context.get("unit_id", "unknown")
    thread_id = context.get("thread_id") or context.get("session_id")

    # Generate UUID if thread_id is missing
    if not thread_id:
        thread_id = str(uuid.uuid4())

    timestamp = int(time.time())

    # Build trace name with optional prefix
    if prefix:
        return f"{prefix}-{user_id}-{unit_id}-{thread_id}-{timestamp}"
    return f"{user_id}-{unit_id}-{thread_id}-{timestamp}"


@contextmanager
def unified_trace_context(
    thread_id: str = None,
    operation_name: str = None,
    custom_name: str = None,
    is_root: bool = False,
):
    """Context manager for unified tracing with a single Weave operation.

    Uses weave.thread() to group all operations under a single thread ID.
    The custom_name becomes the thread_id which shows up in Weave UI.

    Args:
        thread_id: Unique identifier for the trace session
        operation_name: Name of the current operation
        custom_name: Custom display name for the trace (shows in Weave UI)
        is_root: Whether this is the root trace that creates the weave.thread
    """
    if not _trace_enabled:
        yield
        return

    try:
        import weave

        # Generate UUID if thread_id is missing
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # For root traces, establish the weave.thread context
        if is_root:
            # Track this trace as active
            _active_traces[thread_id] = {
                "background_tasks": [],
                "pending_count": 0,
                "start_time": time.time(),
                "attributes": {},
            }
            _trace_locks[thread_id] = threading.Lock()

            try:
                # Use weave.thread with custom_name as the thread_id
                # This groups all child @weave.op calls under this thread
                with weave.thread(thread_id=custom_name or thread_id):
                    try:
                        yield
                    finally:
                        # Wait for background tasks before thread ends
                        _wait_for_trace_completion(thread_id, timeout=10.0)

            finally:
                # Cleanup
                with contextlib.suppress(Exception):
                    if thread_id in _active_traces:
                        del _active_traces[thread_id]
                    if thread_id in _trace_locks:
                        del _trace_locks[thread_id]
        else:
            # Non-root operations just yield within the existing thread context
            yield

    except Exception:
        # Cleanup on error
        if is_root and thread_id in _active_traces:
            with contextlib.suppress(Exception):
                del _active_traces[thread_id]
                del _trace_locks[thread_id]
        raise


def _wait_for_trace_completion(thread_id: str, timeout: float = 5.0):
    """Wait for all background tasks in a trace to complete before session ends.

    This is called within the weave.thread() context to ensure background tasks
    are captured in the trace.

    NOTE: This function no longer blocks for background tasks - background tasks
    should return immediately and complete asynchronously. This is kept for
    backward compatibility but only does a quick check without blocking.

    Args:
        thread_id: The trace session ID
        timeout: Maximum time to wait in seconds (ignored for background tasks)
    """
    if thread_id not in _active_traces:
        return

    # Don't wait for background tasks - they should complete asynchronously
    # Just do a quick non-blocking check to see if there are any pending tasks
    if thread_id not in _trace_locks:
        return

    with _trace_locks[thread_id]:
        trace_info = _active_traces.get(thread_id)
        if trace_info and trace_info["pending_count"] > 0:
            # Background tasks are running - don't block, just return
            # They will complete in the background and update the trace asynchronously
            pass


def register_background_task(thread_id: str, task_id: str):
    """Register a background task for a trace session.

    Args:
        thread_id: The trace session ID
        task_id: Unique ID for the background task
    """
    if thread_id not in _active_traces:
        return

    with _trace_locks[thread_id]:
        trace_info = _active_traces.get(thread_id)
        if trace_info:
            trace_info["background_tasks"].append(task_id)
            trace_info["pending_count"] += 1


def complete_background_task(thread_id: str, task_id: str):
    """Mark a background task as complete for a trace session.

    Args:
        thread_id: The trace session ID
        task_id: Unique ID for the background task
    """
    if thread_id not in _active_traces:
        return

    with _trace_locks[thread_id]:
        trace_info = _active_traces.get(thread_id)
        if trace_info and task_id in trace_info["background_tasks"]:
            trace_info["pending_count"] = max(0, trace_info["pending_count"] - 1)


def add_trace_attributes(attributes: dict[str, Any]) -> None:
    """Add attributes to the current trace.

    Uses weave.attributes() to add metadata to the current operation.

    Args:
        attributes: Dictionary of attributes to add to the current trace
    """
    if not _trace_enabled:
        return

    try:
        import weave

        # Use weave.attributes context manager
        # This adds attributes to the current @weave.op in the call stack
        with weave.attributes(attributes):
            # Attributes are applied to the current trace
            return
    except Exception:
        # Silently fail if attributes can't be added
        pass


def strukt_trace(name: str = None, call_display_name: str = None):
    """Decorator that creates a @weave.op to trace operations.

    This creates a proper @weave.op that will be nested under the weave.thread()
    context established by unified_trace_context.

    NOTE: Currently disabled by default. Use add_trace_attributes() instead
    for simpler attribute-based tracing.
    """

    def decorator(func: F) -> F:
        if not _trace_enabled:
            return func

        try:
            import weave
        except ImportError:
            return func

        # Generate operation name
        if not name:
            class_name = ""
            if hasattr(func, "__qualname__") and "." in func.__qualname__:
                parts = func.__qualname__.split(".")
                if len(parts) > 1:
                    class_name = f".{parts[-2]}"
            op_name = f"StruktX{class_name}.{func.__name__}"
        else:
            op_name = name

        # Apply @weave.op decorator with custom name
        try:
            traced_func = weave.op(
                name=op_name, call_display_name=call_display_name or op_name
            )(func)
            return cast(F, traced_func)
        except Exception:
            # If weave.op fails, return original function
            return func

    return decorator


def init_otel(config: OpenTelemetryConfig) -> None:
    """Initialize an OTLP HTTP exporter to Weave."""
    if not config.enabled:
        return

    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except Exception:
        return

    project_id = config.project_id or os.getenv("WANDB_PROJECT_ID")
    api_key = os.getenv("WANDB_API_KEY")
    if not project_id or not api_key:
        return

    endpoint = _resolve_otlp_endpoint(config)
    headers = _build_otlp_headers(project_id, api_key)

    tracer_provider = trace_sdk.TracerProvider()
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Optional: console exporter when debugging
    if os.getenv("STRUKTX_OTEL_CONSOLE", "0") == "1":
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            tracer_provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
        except Exception:
            pass

    ot_trace.set_tracer_provider(tracer_provider)

    # OpenAI auto-instrumentation
    if config.use_openai_instrumentation:
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        except Exception:
            pass


def _build_otlp_headers(project_id: str, api_key: str) -> dict[str, str]:
    auth = base64.b64encode(f"api:{api_key}".encode()).decode()
    return {
        "Authorization": f"Basic {auth}",
        "project_id": project_id,
    }


def _resolve_otlp_endpoint(config: OpenTelemetryConfig) -> str:
    if config.export_endpoint:
        return config.export_endpoint
    base = os.getenv("WANDB_BASE_URL", "https://trace.wandb.ai")
    if base.endswith("/otel/v1/traces"):
        return base
    if base.endswith("wandb.ai"):
        base = "https://trace.wandb.ai"
    return f"{base}/otel/v1/traces"


def maybe_trace_middleware(tracing_cfg: TracingConfig) -> bool:
    return bool(getattr(tracing_cfg, "enable_middleware_tracing", False))


def enable_global_tracing():
    """Enable global tracing for all StruktX components."""
    global _trace_enabled
    _trace_enabled = True


def disable_global_tracing():
    """Disable global tracing."""
    global _trace_enabled
    _trace_enabled = False


# Auto-instrument all StruktX base classes
def auto_instrument_struktx():
    """Automatically instrument all StruktX base classes and their methods.

    NOTE: Auto-instrumentation is disabled by default to prevent creating
    multiple traces. Use add_trace_attributes() manually in critical paths.
    """
    # Disabled - causes multiple trace creation issues
    # Instead, critical operations should manually call add_trace_attributes()
    pass


def _instrument_class_methods(cls, prefix: str):
    """Instrument all methods of a class with unified tracing.

    NOTE: This is disabled to prevent creating multiple traces.
    """
    # Disabled - causes multiple trace creation issues
    pass
