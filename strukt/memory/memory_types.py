from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid

from pydantic import BaseModel, Field


class KnowledgeCategory(str, Enum):
    """Generic categories for knowledge nodes (extend as needed)."""

    LOCATION = "location"
    PREFERENCE = "preference"
    BEHAVIOR = "behavior"
    CONTEXT = "context"
    RELATIONSHIP = "relationship"
    SCHEDULE = "schedule"
    EMOTION = "emotion"
    FUTURE_EVENT = "future_event"
    OTHER = "other"


class KnowledgeNode(BaseModel):
    """A generic knowledge node suitable for vector storage."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: KnowledgeCategory = Field(default=KnowledgeCategory.OTHER)
    key: str = Field(default="note")
    value: str = Field(default="")
    user_id: Optional[str] = None
    unit_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_id: Optional[str] = None


class KnowledgeEdge(BaseModel):
    """A relationship between two knowledge nodes."""

    source_node_id: str
    target_node_id: str
    relationship: str = Field(default="related_to")
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeStats(BaseModel):
    total_nodes: int = 0
    namespace: Optional[str] = None
    raw_index_info: Dict[str, Any] = Field(default_factory=dict)
