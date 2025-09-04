from __future__ import annotations

import base64
import os
import threading
import weakref
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, cast
import contextlib

from .config import OpenTelemetryConfig, TracingConfig

_T = TypeVar("_T")
F = TypeVar("F", bound=Callable[..., Any])

# Global state for unified tracing
_current_call_stack = threading.local()
_root_call_registry = weakref.WeakValueDictionary()
_trace_enabled = True


def _get_current_call():
    """Get the current call from thread-local storage."""
    return getattr(_current_call_stack, "current_call", None)


def _set_current_call(call):
    """Set the current call in thread-local storage."""
    _current_call_stack.current_call = call


def generate_trace_name(context: dict = None) -> str:
    """Generate a trace name in format: userID-unitID-threadID-timestamp"""
    if not context:
        context = {}

    user_id = context.get("user_id", "unknown")
    unit_id = context.get("unit_id", "unknown")
    thread_id = context.get("thread_id") or context.get("session_id")

    # Generate UUID if thread_id is missing
    if not thread_id:
        thread_id = str(uuid.uuid4())

    timestamp = int(time.time())

    return f"{user_id}-{unit_id}-{thread_id}-{timestamp}"


def _get_or_create_root_call(thread_id: str = None, custom_name: str = None):
    """Get or create a root call for the current thread/session."""
    if not _trace_enabled:
        return None

    try:
        import weave

        # Use thread_id if provided, otherwise generate one
        if not thread_id:
            thread_id = f"struktx_session_{threading.get_ident()}"

        # Check if we already have a root call for this session
        if thread_id in _root_call_registry:
            return _root_call_registry[thread_id]

        # Create display name - use custom_name if provided, otherwise default format
        if custom_name:
            display_name = custom_name
        else:
            display_name = f"Session[{thread_id}]"

        # Create a new root call that will contain everything
        @weave.op(name="StruktX.Session", call_display_name=display_name)
        def _create_root_session():
            return {
                "session_id": thread_id,
                "status": "active",
                "display_name": display_name,
            }

        # Execute the root call to create the session
        _, root_call = _create_root_session.call()

        # Store in registry
        _root_call_registry[thread_id] = root_call

        return root_call
    except Exception:
        return None


@contextmanager
def unified_trace_context(
    thread_id: str = None, operation_name: str = None, custom_name: str = None
):
    """Context manager that ensures all operations are nested under a single root call."""
    if not _trace_enabled:
        yield
        return

    try:
        import weave

        # Generate UUID if thread_id is missing
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Get or create root call
        root_call = _get_or_create_root_call(thread_id, custom_name)
        if not root_call:
            yield
            return

        # Set up thread context if we have a thread_id
        thread_ctx = None
        if thread_id:
            try:
                thread_ctx = weave.thread(thread_id)
            except Exception:
                thread_ctx = None

        # Use the thread context and ensure we're in the root call's context
        ctx_mgr = thread_ctx if thread_ctx else contextlib.nullcontext()

        try:
            with ctx_mgr:
                # Store previous call and set current
                prev_call = _get_current_call()
                _set_current_call(root_call)

                try:
                    yield root_call
                finally:
                    _set_current_call(prev_call)
        except GeneratorExit:
            # Handle generator cleanup gracefully
            pass
        except Exception:
            # Handle any other exceptions gracefully
            yield

    except Exception:
        yield


def strukt_trace(
    name: str = None, call_display_name: str = None, attributes: dict = None
):
    """Decorator that ensures all StruktX operations are nested under the unified trace."""

    def decorator(func: F) -> F:
        if not _trace_enabled:
            return func

        try:
            import weave
        except ImportError:
            return func

        # Generate name if not provided
        if not name:
            class_name = ""
            if hasattr(func, "__qualname__") and "." in func.__qualname__:
                parts = func.__qualname__.split(".")
                if len(parts) > 1:
                    class_name = f".{parts[-2]}"
            op_name = f"StruktX{class_name}.{func.__name__}"
        else:
            op_name = name

        def wrapper(*args, **kwargs):
            try:
                # Try to extract thread_id from various sources
                thread_id = None

                # Check if first arg has context (like InvocationState)
                if (
                    args
                    and hasattr(args[0], "context")
                    and isinstance(args[0].context, dict)
                ):
                    thread_id = args[0].context.get("thread_id") or args[0].context.get(
                        "session_id"
                    )

                # Check kwargs for context
                if not thread_id and "state" in kwargs:
                    state = kwargs["state"]
                    if hasattr(state, "context") and isinstance(state.context, dict):
                        thread_id = state.context.get("thread_id") or state.context.get(
                            "session_id"
                        )

                # Generate UUID if thread_id is missing
                if not thread_id:
                    thread_id = str(uuid.uuid4())

                # Prepare attributes
                call_attrs = dict(attributes or {})
                call_attrs.update(
                    {
                        "struktx.component": True,
                        "struktx.operation": func.__name__,
                        "struktx.thread_id": thread_id,
                    }
                )

                # Execute function and capture result
                result = func(*args, **kwargs)

                # Add result to attributes for better visibility
                if result is not None:
                    try:
                        if hasattr(result, "__dict__"):
                            call_attrs["struktx.result"] = result.__dict__
                        elif isinstance(result, (dict, list, str, int, float, bool)):
                            call_attrs["struktx.result"] = result
                        else:
                            call_attrs["struktx.result"] = str(result)
                    except Exception:
                        call_attrs["struktx.result"] = str(result)

                # Apply attributes to current trace
                with weave.attributes(call_attrs):
                    # Attributes are attached to the current trace context
                    pass

                return result

            except Exception:
                # If anything fails with tracing, just run the function
                return func(*args, **kwargs)

        # Apply weave.op decorator with error handling
        try:
            # Allow dynamic display names via attributes (e.g., component label)
            display_name = call_display_name or op_name
            traced_func = weave.op(name=op_name, call_display_name=display_name)(
                wrapper
            )
            return cast(F, traced_func)
        except Exception:
            # If weave.op fails, return the wrapper without weave.op
            return cast(F, wrapper)

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
    """Automatically instrument all StruktX base classes and their methods."""
    if not _trace_enabled:
        return

    try:
        # Try to import modules, but don't fail if they're not available
        try:
            from . import interfaces
        except ImportError:
            interfaces = None

        try:
            from . import defaults
        except ImportError:
            defaults = None

        # Instrument base interface classes
        if interfaces:
            for cls_name in ["Classifier", "Handler", "MemoryEngine"]:
                try:
                    cls = getattr(interfaces, cls_name, None)
                    if cls and not hasattr(cls, "__subclasshook__"):  # Skip protocols
                        _instrument_class_methods(cls, f"StruktX.{cls_name}")
                except Exception:
                    continue

        # Instrument default implementations
        if defaults:
            for cls_name in [
                "SimpleClassifier",
                "GeneralHandler",
                "SimpleLLMClient",
                "UniversalLLMLogger",
            ]:
                try:
                    cls = getattr(defaults, cls_name, None)
                    if cls:
                        _instrument_class_methods(cls, f"StruktX.{cls_name}")
                except Exception:
                    continue

    except Exception:
        # Silently fail if instrumentation fails
        pass


def _instrument_class_methods(cls, prefix: str):
    """Instrument all methods of a class with unified tracing."""
    try:
        # Get all methods to instrument
        methods_to_trace = []
        try:
            for attr_name in dir(cls):
                if attr_name.startswith("_"):
                    continue

                try:
                    attr = getattr(cls, attr_name)
                    if callable(attr) and not isinstance(
                        attr, (property, staticmethod, classmethod)
                    ):
                        methods_to_trace.append((attr_name, attr))
                except Exception:
                    continue
        except Exception:
            return

        # Instrument each method
        for method_name, method in methods_to_trace:
            try:
                if hasattr(method, "_strukt_traced"):
                    continue  # Already instrumented

                # Create traced version
                traced_method = strukt_trace(
                    name=f"{prefix}.{method_name}",
                    call_display_name=f"{cls.__name__}.{method_name}",
                )(method)

                # Mark as traced and replace
                traced_method._strukt_traced = True
                setattr(cls, method_name, traced_method)

            except Exception:
                # Skip this method if instrumentation fails
                continue

    except Exception:
        pass
