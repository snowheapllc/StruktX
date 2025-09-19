"""
Example demonstrating intent caching functionality.

This example shows how to:
1. Set up intent caching with different strategies
2. Create handlers with caching support
3. Use semantic matching for cache lookups
4. Implement fast-track caching for immediate responses
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone

from ..intent_cache_types import (
    CacheStrategy,
    CacheScope,
    HandlerCacheConfig,
    IntentCacheConfig,
)
from ..intent_cache_engine import InMemoryIntentCacheEngine
from ..handler_cache_mixin import CacheAwareHandler, SemanticCacheHandlerMixin
from ..memory_config import create_default_memory_config
from ...types import HandlerResult, InvocationState


# Example 1: Custom data model for caching
class WeatherCacheData(BaseModel):
    """Custom data model for weather handler caching."""

    location: str
    temperature: float
    condition: str
    humidity: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WeatherHandler(SemanticCacheHandlerMixin, CacheAwareHandler):
    """Example weather handler with semantic caching."""

    def __init__(self):
        super().__init__()
        # Set custom data type for caching
        self.set_cache_data_type(WeatherCacheData)

        # Mock weather data
        self._weather_data = {
            "new york": {"temperature": 22.5, "condition": "sunny", "humidity": 65},
            "london": {"temperature": 15.2, "condition": "cloudy", "humidity": 80},
            "tokyo": {"temperature": 28.1, "condition": "rainy", "humidity": 75},
        }

    def get_cache_config(self) -> HandlerCacheConfig:
        """Custom cache configuration for weather handler."""
        return HandlerCacheConfig(
            handler_name="WeatherHandler",
            cache_data_type=WeatherCacheData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,  # 30 minutes
            max_entries=500,
            similarity_threshold=0.6,  # Reasonable threshold for weather
            enable_fast_track=True,
        )

    def should_cache(self, state: InvocationState, result: HandlerResult) -> bool:
        """Only cache successful weather responses."""
        return result.status == "success" and "weather" in state.text.lower()

    def build_cache_key(self, state: InvocationState) -> str:
        """Build cache key focusing on location and weather intent."""
        # Extract location from text (simplified)
        text = state.text.lower()
        location = "unknown"

        for city in self._weather_data.keys():
            if city in text:
                location = city
                break

        return f"weather:{location}:{state.text}"

    def extract_cache_data(
        self, state: InvocationState, result: HandlerResult
    ) -> WeatherCacheData:
        """Extract weather data for caching."""
        # Parse location from text
        text = state.text.lower()
        location = "unknown"

        for city in self._weather_data.keys():
            if city in text:
                location = city
                break

        # Parse weather data from result
        weather_info = self._weather_data.get(location, {})

        return WeatherCacheData(
            location=location,
            temperature=weather_info.get("temperature", 0.0),
            condition=weather_info.get("condition", "unknown"),
            humidity=weather_info.get("humidity", 0.0),
            metadata={
                "query": state.text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def apply_cached_data(
        self, state: InvocationState, cached_data: WeatherCacheData
    ) -> HandlerResult:
        """Apply cached weather data."""
        response = f"Weather in {cached_data.location}: {cached_data.temperature}°C, {cached_data.condition}, humidity {cached_data.humidity}%"
        return HandlerResult(response=response, status="cached")

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        """Handle weather requests."""
        text = state.text.lower()
        location = "unknown"

        # Find location in text
        for city in self._weather_data.keys():
            if city in text:
                location = city
                break

        if location == "unknown":
            return HandlerResult(
                response="Sorry, I couldn't determine the location. Please specify a city.",
                status="error",
            )

        weather_info = self._weather_data[location]
        response = f"Weather in {location.title()}: {weather_info['temperature']}°C, {weather_info['condition']}, humidity {weather_info['humidity']}%"

        return HandlerResult(response=response, status="success")


class TimeHandler(CacheAwareHandler):
    """Example time handler with both semantic and fast-track caching."""

    def get_cache_config(self) -> HandlerCacheConfig:
        """Cache configuration for time handler."""
        return HandlerCacheConfig(
            handler_name="TimeHandler",
            cache_data_type=dict,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.GLOBAL,  # Time is global
            ttl_seconds=60,  # 1 minute for time
            max_entries=100,
            similarity_threshold=0.9,  # Very high for time queries
            enable_fast_track=True,
        )

    def should_cache(self, state: InvocationState, result: HandlerResult) -> bool:
        """Cache all time responses."""
        return "time" in state.text.lower() or "clock" in state.text.lower()

    def build_cache_key(self, state: InvocationState) -> str:
        """Build cache key for time requests."""
        return f"time:{state.text}"

    def handle(self, state: InvocationState, parts: List[str]) -> HandlerResult:
        """Handle time requests."""
        current_time = datetime.now(timezone.utc)
        response = f"Current time is {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        return HandlerResult(response=response, status="success")


def demonstrate_intent_caching():
    """Demonstrate intent caching functionality."""
    print("=== Intent Caching Demonstration ===\n")

    # 1. Create memory configuration
    config = create_default_memory_config()

    # Customize config for our handlers
    config.update_handler_cache_config(
        "WeatherHandler",
        HandlerCacheConfig(
            handler_name="WeatherHandler",
            cache_data_type=WeatherCacheData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=1800,
            max_entries=500,
            similarity_threshold=0.6,
            enable_fast_track=True,
        ),
    )

    # 2. Create intent cache engine
    cache_engine = InMemoryIntentCacheEngine(config.intent_cache)

    # 3. Create handlers
    weather_handler = WeatherHandler()
    time_handler = TimeHandler()

    # 4. Test weather handler with caching
    print("--- Weather Handler with Semantic Caching ---")

    # First request - should miss cache
    state1 = InvocationState(
        text="What's the weather like in New York?",
        context={"user_id": "user123", "unit_id": "unit456"},
    )
    result1 = weather_handler.handle_with_cache(state1, cache_engine)
    print(f"First request: {result1.response} (status: {result1.status})")

    # Similar request - should hit cache
    state2 = InvocationState(
        text="How's the weather in New York?",
        context={"user_id": "user123", "unit_id": "unit456"},
    )
    result2 = weather_handler.handle_with_cache(state2, cache_engine)
    print(f"Similar request: {result2.response} (status: {result2.status})")

    # Different user - should miss cache
    state3 = InvocationState(
        text="What's the weather in New York?",
        context={"user_id": "user789", "unit_id": "unit456"},
    )
    result3 = weather_handler.handle_with_cache(state3, cache_engine)
    print(f"Different user: {result3.response} (status: {result3.status})")

    # 5. Test time handler with fast-track caching
    print("\n--- Time Handler with Fast-Track Caching ---")

    # First request
    state4 = InvocationState(text="What time is it?")
    result4 = time_handler.handle_with_cache(state4, cache_engine)
    print(f"First time request: {result4.response} (status: {result4.status})")

    # Immediate repeat - should hit fast track
    result5 = time_handler.handle_with_cache(state4, cache_engine)
    print(f"Immediate repeat: {result5.response} (status: {result5.status})")

    # 6. Show cache statistics
    print("\n--- Cache Statistics ---")
    stats = cache_engine.get_stats()
    print(f"Total entries: {stats.total_entries}")
    print(f"Cache hits: {stats.hits}")
    print(f"Cache misses: {stats.misses}")
    print(f"Hit rate: {stats.hit_rate:.2%}")
    print(f"Average similarity: {stats.average_similarity:.2f}")

    # 7. Test cache cleanup
    print("\n--- Cache Cleanup ---")
    cleanup_stats = cache_engine.cleanup()
    print(f"Expired entries removed: {cleanup_stats.expired_entries}")

    # 8. Test cache invalidation
    print("\n--- Cache Invalidation ---")
    invalidated = cache_engine.invalidate("WeatherHandler", user_id="user123")
    print(f"Invalidated {invalidated} entries for user123")


def demonstrate_different_strategies():
    """Demonstrate different caching strategies."""
    print("\n=== Caching Strategies Demonstration ===\n")

    # Create configs for different strategies
    exact_config = IntentCacheConfig(
        enabled=True, default_strategy=CacheStrategy.EXACT, similarity_threshold=1.0
    )

    semantic_config = IntentCacheConfig(
        enabled=True, default_strategy=CacheStrategy.SEMANTIC, similarity_threshold=0.7
    )

    fuzzy_config = IntentCacheConfig(
        enabled=True, default_strategy=CacheStrategy.FUZZY, similarity_threshold=0.8
    )

    # Test different strategies
    strategies = [
        ("Exact", exact_config),
        ("Semantic", semantic_config),
        ("Fuzzy", fuzzy_config),
    ]

    test_queries = [
        "What's the weather in New York?",
        "How's the weather in NYC?",
        "Weather in New York City",
        "What's the temperature in New York?",
    ]

    for strategy_name, config in strategies:
        print(f"--- {strategy_name} Strategy ---")
        engine = InMemoryIntentCacheEngine(config)

        # Store first query
        engine.put(
            "WeatherHandler",
            test_queries[0],
            {"response": "Sunny, 22°C"},
            user_id="test_user",
        )

        # Test matches
        for i, query in enumerate(test_queries):
            match = engine.get("WeatherHandler", query, user_id="test_user")
            if match:
                print(
                    f"  Query {i + 1}: MATCH (similarity: {match.similarity_score:.2f})"
                )
            else:
                print(f"  Query {i + 1}: NO MATCH")
        print()


if __name__ == "__main__":
    demonstrate_intent_caching()
    demonstrate_different_strategies()
