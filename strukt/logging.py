from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.json import JSON as RichJSON
from rich.traceback import install as rich_tracebacks


_GLOBAL_CONSOLE: Console | None = None


def _get_console() -> Console:
    global _GLOBAL_CONSOLE
    if _GLOBAL_CONSOLE is None:
        # Auto width, colorful output; end-user can disable with STRUKTX_NO_RICH=1
        _GLOBAL_CONSOLE = Console(stderr=False)
        if os.getenv("STRUKTX_RICH_TRACEBACK", "1") != "0":
            rich_tracebacks(show_locals=False)
    return _GLOBAL_CONSOLE


def format_prompt_preserving_json(template: str, variables: Dict[str, Any]) -> str:
    """Safely replace placeholders like {name} without breaking JSON examples.

    This performs a conservative literal replacement for exact tokens of the form
    '{key}', leaving any other braces untouched (including JSON braces). This avoids
    Python str.format() pitfalls with unescaped braces inside code examples.
    """
    result = template
    for key, value in variables.items():
        token = "{" + str(key) + "}"
        result = result.replace(token, str(value))
    return result


def render_prompt_with_safe_braces(template: str, variables: Dict[str, Any]) -> str:
    """Render a prompt where only known variables are formatted.

    Algorithm:
    1) Escape every brace by doubling it to neutralize formatting.
    2) For each variable key, revert '{{key}}' back to '{key}'.
    3) Call str.format(**variables) to substitute only allowed variables.

    This preserves JSON examples and any other brace usage while still allowing
    specific placeholders to be replaced. It avoids 'missing variables' errors
    in PromptTemplate-style engines encountering stray braces.
    """
    # Step 1: escape all braces
    escaped = template.replace("{", "{{").replace("}", "}}")
    # Step 2: un-escape allowed placeholders
    for key in variables.keys():
        escaped = escaped.replace("{{" + str(key) + "}}", "{" + str(key) + "}")
    # Step 3: format
    try:
        return escaped.format(**variables)
    except Exception:
        # Fallback to conservative replacer if format fails for any reason
        return format_prompt_preserving_json(template, variables)


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
