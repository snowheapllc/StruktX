from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Any, List, Tuple

from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as rich_tracebacks

from .middleware import Middleware
from .types import HandlerResult, InvocationState, QueryClassification

_GLOBAL_CONSOLE: Console | None = None


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


class LoggingMiddleware(Middleware):
    def __init__(self, *, verbose: bool = False) -> None:
        self._verbose = verbose
        self.console = Console()

    def before_classify(self, state: InvocationState) -> InvocationState:
        if self._verbose:
            panel = Panel(
                f"[cyan]Text:[/cyan] {state.text}\n[dim]Context:[/dim] {list(state.context.keys())}",
                title="🔍 [bold blue]Logging: Before Classify[/bold blue]",
                border_style="blue",
            )
            self.console.print(panel)
        return state

    def after_classify(
        self,
        state: InvocationState,
        classification: QueryClassification,
    ) -> Tuple[InvocationState, QueryClassification]:
        table = Table(
            title="🎯 [bold green]Logging: Classification[/bold green]",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Type", style="cyan")
        table.add_column("Confidence", style="yellow")
        table.add_column("Part", style="magenta")

        for qtype, conf, part in zip(
            classification.query_types, classification.confidences, classification.parts
        ):
            table.add_row(
                qtype, f"{conf:.2f}", part[:40] + "..." if len(part) > 40 else part
            )

        self.console.print(table)
        return state, classification

    def before_handle(
        self, state: InvocationState, query_type: str, parts: List[str]
    ) -> Tuple[InvocationState, List[str]]:
        if self._verbose:
            panel = Panel(
                f"[cyan]Type:[/cyan] {query_type}\n[cyan]Parts:[/cyan] {parts}",
                title="⚙️ [bold yellow]Logging: Before Handle[/bold yellow]",
                border_style="yellow",
            )
            self.console.print(panel)
        return state, parts

    def after_handle(
        self, state: InvocationState, query_type: str, result: HandlerResult
    ) -> HandlerResult:
        panel = Panel(
            f"[cyan]Status:[/cyan] {result.status}\n[cyan]Response:[/cyan] {result.response}",
            title="✅ [bold green]Logging: After Handle[/bold green]",
            border_style="green",
        )
        self.console.print(panel)
        return result
