"""Logging middleware for structured logging of handler operations."""

from __future__ import annotations

from typing import List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from strukt.middleware import Middleware
from strukt.types import HandlerResult, InvocationState, QueryClassification


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
