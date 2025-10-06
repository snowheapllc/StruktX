from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class StruktQueryEnum:
    GENERAL = "general"
    MULTIPLE = "multiple"
    QUERY = "query"
    COMMAND = "command"
    ACTION = "action"
    INFORMATION = "information"
    ERROR = "error"


@dataclass
class InvocationState:
    """State passed through classification and handlers.

    This is intentionally generic for framework portability. Users can put
    any metadata in `context`.
    """

    text: str
    context: dict[str, Any] = field(default_factory=lambda: {"handler_intents": {}})
    query_types: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)


@dataclass
class QueryClassification:
    query_types: list[str]
    confidences: list[float]
    parts: list[str]


@dataclass
class HandlerResult:
    response: str
    status: str


@dataclass
class StruktResponse:
    response: str | dict[str, Any]
    query_type: str
    query_types: list[str]
    parts: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class BackgroundTaskResult(TypedDict):
    """Result of a background task."""

    response: str | dict[str, Any]
    status: str


class BackgroundTaskInfo(TypedDict):
    """Information about a background task."""

    task_id: str
    handler_name: str
    handler_id: str
    status: str
    progress: float
    created_at: str
    started_at: str | None
    completed_at: str | None
    result: BackgroundTaskResult | None
    error: str | None
    metadata: dict[str, Any]
