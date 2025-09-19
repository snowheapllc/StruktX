from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, Type

from .intent_cache_types import (
    HandlerCache,
    HandlerCacheConfig,
    CacheStrategy,
    CacheScope,
)
from ..types import HandlerResult, InvocationState
from ..logging import get_logger


class HandlerCacheMixin(HandlerCache, ABC):
    """Mixin class that provides default implementations for handler caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_config: Optional[HandlerCacheConfig] = None

    def get_cache_config(self) -> HandlerCacheConfig:
        """Return the cache configuration for this handler."""
        if self._cache_config is None:
            self._cache_config = self._create_default_cache_config()
        return self._cache_config

    def _create_default_cache_config(self) -> HandlerCacheConfig:
        """Create default cache configuration. Override in subclasses for customization."""
        return HandlerCacheConfig(
            handler_name=self.__class__.__name__,
            cache_data_type=dict,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            max_entries=1000,
            similarity_threshold=0.7,
            enable_fast_track=True,
        )

    def should_cache(self, state: InvocationState, result: HandlerResult) -> bool:
        """Default implementation: cache successful results."""
        return result.status == "success" or result.status == "cached"

    def build_cache_key(self, state: InvocationState) -> str:
        """Build cache key from invocation state."""
        # Include text, user context, and relevant metadata
        key_parts = [state.text]

        # Add user/unit context if available
        if state.context.get("user_id"):
            key_parts.append(f"user:{state.context['user_id']}")
        if state.context.get("unit_id"):
            key_parts.append(f"unit:{state.context['unit_id']}")

        # Add query types for more specific caching
        if state.query_types:
            key_parts.append(f"types:{','.join(state.query_types)}")

        # Add parts if available
        if state.parts:
            key_parts.append(f"parts:{','.join(state.parts)}")

        return "|".join(key_parts)

    def extract_cache_data(
        self, state: InvocationState, result: HandlerResult
    ) -> Dict[str, Any]:
        """Extract data to be cached. Override for custom data extraction."""
        return {
            "response": result.response,
            "status": result.status,
            "query_types": state.query_types,
            "parts": state.parts,
            "timestamp": state.context.get("timestamp"),
            "metadata": state.context.get("metadata", {}),
        }

    def apply_cached_data(
        self, state: InvocationState, cached_data: Dict[str, Any]
    ) -> HandlerResult:
        """Apply cached data to generate a result."""
        return HandlerResult(response=cached_data.get("response", ""), status="cached")


class FastTrackHandlerMixin:
    """Mixin for handlers that want to implement fast-track caching."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fast_track_cache: Dict[str, Any] = {}
        self._fast_track_config = {
            "max_entries": 100,
            "ttl_seconds": 300,  # 5 minutes for fast track
            "enabled": True,
        }
        self._logger = get_logger("fast_track_cache")

    def get_fast_track_key(self, state: InvocationState) -> str:
        """Generate a fast track cache key."""
        return f"fast:{hash(state.text)}:{state.context.get('user_id', '')}:{state.context.get('unit_id', '')}"

    def get_fast_track(self, state: InvocationState) -> Optional[HandlerResult]:
        """Get fast track cached result."""
        if not self._fast_track_config["enabled"]:
            return None

        key = self.get_fast_track_key(state)
        if key in self._fast_track_cache:
            cached_data = self._fast_track_cache[key]
            # Log fast track hit with JSON format
            self._logger.cache_result(
                title="Fast Track Cache Hit",
                data={
                    "fast_track_hit": True,
                    "handler_name": self.__class__.__name__,
                    "key": key[:30] + "..." if len(key) > 30 else key,
                },
            )
            return HandlerResult(
                response=cached_data["response"], status="fast_track_cached"
            )
        return None

    def set_fast_track(self, state: InvocationState, result: HandlerResult) -> None:
        """Set fast track cached result."""
        if not self._fast_track_config["enabled"]:
            return

        key = self.get_fast_track_key(state)
        self._fast_track_cache[key] = {
            "response": result.response,
            "status": result.status,
            "timestamp": state.context.get("timestamp"),
        }

        # Log fast track store with JSON format
        self._logger.cache_result(
            title="Fast Track Cache Store",
            data={
                "fast_track_store": True,
                "handler_name": self.__class__.__name__,
                "key": key[:30] + "..." if len(key) > 30 else key,
            },
        )

        # Simple cleanup - remove oldest entries if over limit
        if len(self._fast_track_cache) > self._fast_track_config["max_entries"]:
            # Remove first (oldest) entry
            oldest_key = next(iter(self._fast_track_cache))
            del self._fast_track_cache[oldest_key]

    def clear_fast_track(self) -> None:
        """Clear all fast track cache entries."""
        self._fast_track_cache.clear()


class SemanticCacheHandlerMixin(HandlerCacheMixin):
    """Mixin for handlers that want semantic caching with custom data models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_data_type: Optional[Type] = None

    def set_cache_data_type(self, data_type: Type) -> None:
        """Set the Pydantic model type for cached data."""
        self._cache_data_type = data_type

    def get_cache_config(self) -> HandlerCacheConfig:
        """Get cache config with custom data type if set."""
        config = super().get_cache_config()
        if self._cache_data_type:
            config.cache_data_type = self._cache_data_type
        return config

    def extract_cache_data(self, state: InvocationState, result: HandlerResult) -> Any:
        """Extract data using the configured data type."""
        if self._cache_data_type:
            # Create instance of the configured data type
            return self._cache_data_type(
                response=result.response,
                status=result.status,
                query_types=state.query_types,
                parts=state.parts,
                metadata=state.context.get("metadata", {}),
            )
        else:
            # Fall back to dict
            return super().extract_cache_data(state, result)


class CacheAwareHandler(HandlerCacheMixin, FastTrackHandlerMixin):
    """Base class for handlers that want both semantic and fast-track caching."""

    def handle_with_cache(
        self, state: InvocationState, cache_engine: Optional[Any] = None
    ) -> HandlerResult:
        """Handle with caching support."""
        # Try fast track first
        fast_track_result = self.get_fast_track(state)
        if fast_track_result:
            return fast_track_result

        # Try semantic cache
        if cache_engine:
            cache_key = self.build_cache_key(state)
            cache_match = cache_engine.get(
                handler_name=self.__class__.__name__,
                cache_key=cache_key,
                user_id=state.context.get("user_id"),
                unit_id=state.context.get("unit_id"),
                session_id=state.context.get("session_id"),
            )

            if cache_match and cache_match.is_valid:
                # Apply cached data
                result = self.apply_cached_data(state, cache_match.entry.cached_data)
                # Also set in fast track for next time
                self.set_fast_track(state, result)
                return result

        # No cache hit, execute handler using super().handle to avoid recursion
        result = super().handle(state, state.parts)

        # Cache the result if appropriate
        if self.should_cache(state, result):
            # Set fast track
            self.set_fast_track(state, result)

            # Set semantic cache
            if cache_engine:
                cache_key = self.build_cache_key(state)
                cache_data = self.extract_cache_data(state, result)
                cache_engine.put(
                    handler_name=self.__class__.__name__,
                    cache_key=cache_key,
                    data=cache_data,
                    user_id=state.context.get("user_id"),
                    unit_id=state.context.get("unit_id"),
                    session_id=state.context.get("session_id"),
                )

        return result
