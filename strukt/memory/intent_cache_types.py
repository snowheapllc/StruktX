from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, Optional, Type, TypeVar
from ..types import HandlerResult, InvocationState
import uuid

from pydantic import BaseModel, Field


class DictData(BaseModel):
    """Simple dict-like data model for caching."""

    data: Dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def items(self) -> Any:
        return self.data.items()

    def keys(self) -> Any:
        return self.data.keys()

    def values(self) -> Any:
        return self.data.values()


class CacheStrategy(str, Enum):
    """Strategies for cache matching and invalidation."""

    EXACT = "exact"  # Exact string match
    SEMANTIC = "semantic"  # Semantic similarity match
    FUZZY = "fuzzy"  # Fuzzy string matching
    HYBRID = "hybrid"  # Combination of strategies


class CacheScope(str, Enum):
    """Scope for cache entries."""

    GLOBAL = "global"  # Available to all users/units
    USER = "user"  # Scoped to specific user
    UNIT = "unit"  # Scoped to specific unit
    SESSION = "session"  # Scoped to current session


T = TypeVar("T", bound=BaseModel)


class IntentCacheEntry(BaseModel, Generic[T]):
    """A cached intent with associated data and metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    handler_name: str
    cache_key: str  # The hashed key used for exact matching
    original_text: str  # The original text for semantic matching
    cached_data: T  # The actual cached data (Pydantic model)
    user_id: Optional[str] = None
    unit_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(default=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ttl_seconds: Optional[int] = None  # Time to live in seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1


class CacheMatch(BaseModel):
    """Result of a cache lookup with match quality information."""

    entry: IntentCacheEntry[Any]
    similarity_score: float = Field(ge=0.0, le=1.0)
    match_type: CacheStrategy
    is_exact: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if the match is valid (not expired and meets confidence threshold)."""
        return not self.entry.is_expired() and self.similarity_score >= 0.7


class HandlerCacheConfig(BaseModel):
    """Configuration for handler-specific caching."""

    handler_name: str
    cache_data_type: Type[BaseModel] = (
        DictData  # Default to DictData, but allow other BaseModel subclasses
    )
    strategy: CacheStrategy = CacheStrategy.SEMANTIC
    scope: CacheScope = CacheScope.USER
    ttl_seconds: Optional[int] = 3600  # 1 hour default
    max_entries: int = 1000
    similarity_threshold: float = 0.7
    enable_fast_track: bool = True  # Enable fast track for exact matches
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntentCacheConfig(BaseModel):
    """Global configuration for intent caching."""

    enabled: bool = True
    default_strategy: CacheStrategy = CacheStrategy.SEMANTIC
    default_ttl_seconds: int = 3600
    max_global_entries: int = 10000
    similarity_threshold: float = 0.7
    cleanup_interval_seconds: int = 300  # 5 minutes
    handler_configs: Dict[str, HandlerCacheConfig] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CacheStats(BaseModel):
    """Statistics about cache performance."""

    total_entries: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired_entries: int = 0
    hit_rate: float = 0.0
    average_similarity: float = 0.0
    handler_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class HandlerCache(ABC):
    """Interface for handlers that want to implement internal caching."""

    @abstractmethod
    def get_cache_config(self) -> HandlerCacheConfig:
        """Return the cache configuration for this handler."""
        pass

    @abstractmethod
    def should_cache(self, state: InvocationState, result: HandlerResult) -> bool:
        """Determine if the result should be cached."""
        pass

    @abstractmethod
    def build_cache_key(self, state: InvocationState) -> str:
        """Build a cache key from the invocation state."""
        pass

    @abstractmethod
    def extract_cache_data(
        self, state: InvocationState, result: HandlerResult
    ) -> BaseModel:
        """Extract data to be cached from the state and result."""
        pass

    def apply_cached_data(
        self, state: InvocationState, cached_data: BaseModel
    ) -> HandlerResult:
        """Apply cached data to generate a result (optional override)."""
        # Default implementation - handlers can override for custom logic
        return HandlerResult(response=cached_data.model_dump_json(), status="cached")


class IntentCacheEngine(ABC):
    """Abstract base class for intent cache engines."""

    @abstractmethod
    def get(
        self,
        handler_name: str,
        cache_key: str,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[CacheMatch]:
        """Get a cached entry with semantic matching."""
        pass

    @abstractmethod
    def put(
        self,
        handler_name: str,
        cache_key: str,
        data: BaseModel,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Store data in the cache."""
        pass

    @abstractmethod
    def invalidate(
        self,
        handler_name: str,
        cache_key: Optional[str] = None,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries."""
        pass

    @abstractmethod
    def cleanup(self) -> CacheStats:
        """Clean up expired entries and return statistics."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        pass
