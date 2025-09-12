from __future__ import annotations

from datetime import datetime
import json
import os
import contextlib
from typing import Any, Optional, Callable, TypeVar, cast

from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.traceback import install as rich_tracebacks

_GLOBAL_CONSOLE: Console | None = None

# Type variable for function decorators
F = TypeVar("F", bound=Callable[..., Any])

# Global Weave state - shared across all logger instances
_global_weave_initialized = False
_global_weave_project_name = None
_global_weave_environment = None


def _get_console() -> Console:
    global _GLOBAL_CONSOLE
    if _GLOBAL_CONSOLE is None:
        # Auto width, colorful output; end-user can disable with STRUKTX_NO_RICH=1
        _GLOBAL_CONSOLE = Console(stderr=False)
        if os.getenv("STRUKTX_RICH_TRACEBACK", "1") != "0":
            rich_tracebacks(show_locals=False)
    return _GLOBAL_CONSOLE


class StruktLogger:
    def __init__(self, name: str = "struktx") -> None:
        self._name = name
        self._console = _get_console()
        # Levels: debug <= info <= warn <= error
        self._level = os.getenv("STRUKTX_LOG_LEVEL", "info").lower()

        # Weave integration - use global state
        self._weave_initialized = _global_weave_initialized
        self._weave_project_name = _global_weave_project_name
        self._weave_environment = _global_weave_environment

    def _init_weave(
        self, project_name: Optional[str] = None, environment: Optional[str] = None
    ) -> None:
        """Initialize Weave logging if available and enabled."""
        if self._weave_initialized:
            return

        try:
            import weave

            # Check if WANDB_API_KEY is available
            api_key = os.getenv("WANDB_API_KEY")
            if not api_key:
                self.warn(
                    "WANDB_API_KEY environment variable not set - Weave logging disabled"
                )
                return

            # Get project name and environment from parameters or environment variables
            self._weave_project_name = project_name or os.getenv(
                "PROJECT_NAME", "struktx"
            )
            self._weave_environment = environment or os.getenv(
                "CURRENT_ENV", "development"
            )

            # Initialize Weave with the project - disable autopatching to prevent LangChain noise
            weave.init(
                project_name=f"{self._weave_project_name}-{self._weave_environment}",
                autopatch_settings={"disable_autopatch": True},
            )
            # Update global state
            global \
                _global_weave_initialized, \
                _global_weave_project_name, \
                _global_weave_environment
            _global_weave_initialized = True
            _global_weave_project_name = self._weave_project_name
            _global_weave_environment = self._weave_environment

            # Update instance state
            self._weave_initialized = True
            self.info(
                f"Weave logging initialized for project: {self._weave_project_name}-{self._weave_environment}"
            )

        except ImportError:
            self.warn("Weave not available. Install with: pip install weave")
        except Exception as e:
            self.warn(f"Failed to initialize Weave: {e}")
            import traceback

            self.warn(f"Weave init traceback: {traceback.format_exc()}")

    def init_weave(
        self, project_name: Optional[str] = None, environment: Optional[str] = None
    ) -> None:
        """Public method to initialize Weave logging."""
        self._init_weave(project_name, environment)

    def is_weave_available(self) -> bool:
        """Check if Weave is available and initialized."""
        global _global_weave_initialized
        return _global_weave_initialized

    def get_weave_project_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get the current Weave project name and environment."""
        return self._weave_project_name, self._weave_environment

    @contextlib.contextmanager
    def weave_context(
        self,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        unit_name: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """Context manager for Weave logging with user context.

        Args:
            user_id: Explicit user ID (takes precedence over context)
            unit_id: Explicit unit ID (takes precedence over context)
            unit_name: Explicit unit name (takes precedence over context)
            context: Dictionary to extract user context from (fallback if explicit values not provided)
        """
        if not self._weave_initialized:
            yield None
            return

        try:
            import weave

            # Extract values from context if not explicitly provided
            final_user_id = user_id or (context.get("user_id") if context else None)
            final_unit_id = unit_id or (context.get("unit_id") if context else None)
            final_unit_name = unit_name or (
                context.get("unit_name") if context else None
            )

            # Build attributes dictionary, only including non-None values
            attributes = {}
            if final_user_id:
                attributes["user_id"] = final_user_id
            if final_unit_id:
                attributes["unit_id"] = final_unit_id
            if final_unit_name:
                attributes["unit_name"] = final_unit_name

            # Use weave.attributes context manager to add user context to all calls within this context
            with weave.attributes(attributes):
                yield
        except ImportError:
            # If weave is not available, yield None
            yield None
        except Exception as e:
            # Log any Weave-related errors but don't fail the main operation
            self.warn(f"Weave context setup failed: {e}")
            yield None

    @contextlib.contextmanager
    def weave_context_from_state(self, state):
        """Context manager that automatically extracts user context from InvocationState.

        Args:
            state: InvocationState object containing context information
        """
        context = getattr(state, "context", {}) if state else {}
        with self.weave_context(context=context):
            yield

    def create_weave_op(
        self,
        func: Optional[F] = None,
        name: Optional[str] = None,
        call_display_name: Optional[str] = None,
    ) -> F:
        """Create a Weave operation decorator for tracking function calls.

        This method creates a Weave operation that will automatically track:
        - Function inputs and outputs
        - Execution time
        - User context (user_id, unit_id, unit_name) when used within weave_context

        Args:
            func: The function to decorate (used internally by the decorator)
            name: Optional custom name for the operation
            call_display_name: Optional custom display name for calls

        Returns:
            Decorated function with Weave tracking
        """
        if not self._weave_initialized:
            # If Weave is not available, return the original function unchanged
            if func is None:
                return cast(F, lambda f: f)
            else:
                return func

        try:
            import weave

            def decorator(f: F) -> F:
                # Create the Weave operation
                op = weave.op(name=name, call_display_name=call_display_name)(f)
                return cast(F, op)

            # Handle both @create_weave_op and @create_weave_op() usage
            if func is None:
                return cast(F, decorator)
            else:
                return decorator(func)

        except ImportError:
            # If Weave is not available, return the original function unchanged
            if func is None:
                return cast(F, lambda f: f)
            else:
                return func

    def _should(self, level: str) -> bool:
        order = {"debug": 10, "info": 20, "warn": 30, "error": 40}
        return order.get(level, 20) >= order.get(self._level, 20)

    def _stamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _maybe_truncate(self, text: str) -> str:
        try:
            max_len = int(os.getenv("STRUKTX_LOG_MAXLEN", "500"))
        except Exception:
            max_len = 4000
        if os.getenv("STRUKTX_LOG_VERBOSE") == "1":
            return text
        if len(text) <= max_len:
            return text
        return text[:max_len] + "\n[...truncated...]"

    def debug(self, message: str) -> None:
        if not self._should("debug"):
            return
        panel = Panel(
            self._maybe_truncate(message),
            title=f"🐞 [bold cyan]{self._name}[/bold cyan] DEBUG",
            border_style="cyan",
        )
        self._console.print(panel)

    def info(self, message: str) -> None:
        if not self._should("info"):
            return
        panel = Panel(
            self._maybe_truncate(message),
            title=f"ℹ️ [bold blue]{self._name}[/bold blue] INFO",
            border_style="blue",
        )
        self._console.print(panel)

    def memory_injection(self, source: str, count: int) -> None:
        """Log memory injection with source context."""
        if not self._should("info"):
            return
        message = f"Injecting {count} memory item(s) into prompt"
        panel = Panel(
            self._maybe_truncate(message),
            # Use angle brackets instead of square brackets to avoid Rich markup parsing
            title=f"🧠 [bold magenta]{self._name}[/bold magenta] MEMORY <{source}>",
            border_style="magenta",
        )
        self._console.print(panel)

    def warn(self, message: str) -> None:
        if not self._should("warn"):
            return
        panel = Panel(
            self._maybe_truncate(message),
            title=f"⚠️ [bold yellow]{self._name}[/bold yellow] WARN",
            border_style="yellow",
        )
        self._console.print(panel)

    def warning(self, message: str) -> None:
        if not self._should("warn"):
            return
        panel = Panel(
            self._maybe_truncate(message),
            title=f"⚠️ [bold yellow]{self._name}[/bold yellow] WARN",
            border_style="yellow",
        )
        self._console.print(panel)

    def error(self, message: str) -> None:
        if not self._should("error"):
            return
        panel = Panel(
            self._maybe_truncate(message),
            title=f"🛑 [bold red]{self._name}[/bold red] ERROR",
            border_style="red",
        )
        self._console.print(panel)

    def exception(self, message: str) -> None:
        # On exception, do not truncate
        panel = Panel(
            message,
            title=f"💥 [bold red]{self._name}[/bold red] EXCEPTION",
            border_style="red",
        )
        self._console.print(panel)

    def prompt_template(self, title: str, template: str) -> None:
        panel = Panel(
            self._maybe_truncate(template),
            title=f"🧩 Prompt Template: {title}",
            border_style="magenta",
        )
        self._console.print(panel)

    def prompt(self, title: str, content: str) -> None:
        panel = Panel(
            self._maybe_truncate(content),
            title=f"📝 Prompt: {title}",
            border_style="green",
        )
        self._console.print(panel)

    def json(self, title: str, data: Any) -> None:
        try:
            rendered = RichJSON.from_data(data)
            panel = Panel(rendered, title=f"📦 JSON: {title}", border_style="cyan")
            self._console.print(panel)
        except Exception:
            # Fallback to plain dump
            text = json.dumps(data, indent=2, ensure_ascii=False)
            panel = Panel(
                self._maybe_truncate(text),
                title=f"📦 JSON (raw): {title}",
                border_style="cyan",
            )
            self._console.print(panel)


def get_logger(name: str = "struktx") -> StruktLogger:
    return StruktLogger(name)
