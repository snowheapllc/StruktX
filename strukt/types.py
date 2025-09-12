from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict, Union


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
    context: Dict[str, Any] = field(default_factory=lambda: {"handler_intents": {}})
    query_types: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    parts: List[str] = field(default_factory=list)


@dataclass
class QueryClassification:
    query_types: List[str]
    confidences: List[float]
    parts: List[str]


@dataclass
class HandlerResult:
    response: str
    status: str


@dataclass
class StruktResponse:
    response: Union[str, Dict[str, Any]]
    query_type: str
    query_types: List[str]
    parts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackgroundTaskResult(TypedDict):
    """Result of a background task."""

    response: Union[str, Dict[str, Any]]
    status: str


class BackgroundTaskInfo(TypedDict):
    """Information about a background task."""

    task_id: str
    handler_name: str
    handler_id: str
    status: str
    progress: float
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[BackgroundTaskResult]
    error: Optional[str]
    metadata: Dict[str, Any]
