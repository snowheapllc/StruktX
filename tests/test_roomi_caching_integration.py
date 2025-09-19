"""
Test Roomi caching integration without importing the actual handlers.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from strukt.types import InvocationState, HandlerResult
from strukt.memory import InMemoryIntentCacheEngine, create_default_memory_config

from strukt.extensions.roomi.cached_handlers import (
    WeatherCacheData,
    FutureEventCacheData,
    HelpdeskCacheData,
    EventCacheData,
    BillCacheData,
    NotificationCacheData,
    AmenityCacheData,
)


class TestRoomiCachingIntegration:
    """Test Roomi caching integration functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create cache engine
        memory_config = create_default_memory_config()
        self.cache_engine = InMemoryIntentCacheEngine(memory_config.intent_cache)
    
    def test_weather_cache_data_model(self):
        """Test weather cache data model."""
        from datetime import datetime, timezone
        
        cache_data = WeatherCacheData(
            location="Dubai",
            weather_data={
                "temperature": 25.0,
                "condition": "sunny",
                "humidity": 60
            },
            message="Weather data retrieved",
            status="success",
            current_date="2024-01-01",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        assert cache_data.location == "Dubai"
        assert cache_data.weather_data["temperature"] == 25.0
        assert cache_data.status == "success"
        assert cache_data.message == "Weather data retrieved"
    
    def test_future_event_cache_data_model(self):
        """Test future event cache data model."""
        from datetime import datetime, timezone
        
        cache_data = FutureEventCacheData(
            event_data={
                "event_id": "evt_123",
                "title": "Test Event",
                "date": "2024-01-01"
            },
            message="Event created successfully",
            status="success",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        assert cache_data.event_data["event_id"] == "evt_123"
        assert cache_data.status == "success"
        assert cache_data.message == "Event created successfully"
    
    def test_helpdesk_cache_data_model(self):
        """Test helpdesk cache data model."""
        from datetime import datetime, timezone
        
        cache_data = HelpdeskCacheData(
            ticket_data={
                "ticket_id": "tkt_123",
                "title": "Test Issue",
                "status": "open"
            },
            message="Ticket created successfully",
            status="success",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        assert cache_data.ticket_data["ticket_id"] == "tkt_123"
        assert cache_data.status == "success"
        assert cache_data.message == "Ticket created successfully"
    
    def test_cache_engine_basic_operations(self):
        """Test basic cache engine operations."""
        # Test storing weather data
        weather_data = WeatherCacheData(
            location="Dubai",
            weather_data={"temperature": 25.0, "condition": "sunny"},
            message="Weather data retrieved",
            status="success",
            current_date="2024-01-01",
            timestamp="2024-01-01T10:00:00Z"
        )
        
        entry_id = self.cache_engine.put(
            "WeatherHandler",
            "weather:dubai:What's the weather like in Dubai?",
            weather_data,
            user_id="user123",
            unit_id="unit456"
        )
        
        assert entry_id != ""
        
        # Test retrieving weather data
        match = self.cache_engine.get(
            "WeatherHandler",
            "weather:dubai:How's the weather in Dubai?",
            user_id="user123",
            unit_id="unit456"
        )
        
        assert match is not None
        assert match.similarity_score > 0.5
        assert match.entry.cached_data.location == "Dubai"
        assert match.entry.cached_data.weather_data["temperature"] == 25.0
    
    def test_cache_engine_semantic_matching(self):
        """Test semantic matching in cache engine."""
        # Store event data
        event_data = FutureEventCacheData(
            event_data={"event_id": "evt_123", "title": "Meeting"},
            message="Event created",
            status="success",
            timestamp="2024-01-01T10:00:00Z"
        )
        
        self.cache_engine.put(
            "FutureEventHandler",
            "future_event:Schedule a meeting for tomorrow",
            event_data,
            user_id="user123",
            unit_id="unit456"
        )
        
        # Test semantic matching with similar queries
        similar_queries = [
            "future_event:Create a meeting for tomorrow",
            "future_event:Set up a meeting for tomorrow",
            "future_event:Book a meeting for tomorrow",
        ]
        
        for query in similar_queries:
            match = self.cache_engine.get(
                "FutureEventHandler",
                query,
                user_id="user123",
                unit_id="unit456"
            )
            assert match is not None, f"Should match: {query}"
            assert match.similarity_score > 0.3, f"Similarity too low: {match.similarity_score}"
    
    def test_cache_engine_user_scoping(self):
        """Test that cache entries are scoped by user."""
        # Store data for user1
        weather_data = WeatherCacheData(
            location="Dubai",
            weather_data={"temperature": 25.0},
            message="Weather data",
            status="success",
            current_date="2024-01-01",
            timestamp="2024-01-01T10:00:00Z"
        )
        
        self.cache_engine.put(
            "WeatherHandler",
            "weather:dubai:What's the weather in Dubai?",
            weather_data,
            user_id="user1",
            unit_id="unit1"
        )
        
        # Test retrieval for same user - should match
        match1 = self.cache_engine.get(
            "WeatherHandler",
            "weather:dubai:How's the weather in Dubai?",
            user_id="user1",
            unit_id="unit1"
        )
        assert match1 is not None
        
        # Test retrieval for different user - should not match
        match2 = self.cache_engine.get(
            "WeatherHandler",
            "weather:dubai:How's the weather in Dubai?",
            user_id="user2",
            unit_id="unit1"
        )
        assert match2 is None
    
    def test_cache_engine_statistics(self):
        """Test cache engine statistics."""
        # Store some data
        weather_data = WeatherCacheData(
            location="Dubai",
            weather_data={"temperature": 25.0},
            message="Weather data",
            status="success",
            current_date="2024-01-01",
            timestamp="2024-01-01T10:00:00Z"
        )
        
        self.cache_engine.put(
            "WeatherHandler",
            "weather:dubai:What's the weather in Dubai?",
            weather_data,
            user_id="user123",
            unit_id="unit456"
        )
        
        # Test retrieval - should hit
        match = self.cache_engine.get(
            "WeatherHandler",
            "weather:dubai:How's the weather in Dubai?",
            user_id="user123",
            unit_id="unit456"
        )
        
        # Get statistics
        stats = self.cache_engine.get_stats()
        assert stats.total_entries == 1
        assert stats.hits >= 1
        assert stats.misses >= 0
        assert stats.hit_rate > 0
        assert "WeatherHandler" in stats.handler_stats
    
    def test_cache_engine_cleanup(self):
        """Test cache engine cleanup."""
        # Store some data
        weather_data = WeatherCacheData(
            location="Dubai",
            weather_data={"temperature": 25.0},
            message="Weather data",
            status="success",
            current_date="2024-01-01",
            timestamp="2024-01-01T10:00:00Z"
        )
        
        self.cache_engine.put(
            "WeatherHandler",
            "weather:dubai:What's the weather in Dubai?",
            weather_data,
            user_id="user123",
            unit_id="unit456",
            ttl_seconds=1  # Very short TTL
        )
        
        # Wait a bit and cleanup
        import time
        time.sleep(1.1)
        
        stats = self.cache_engine.cleanup()
        assert stats.expired_entries >= 1
        assert stats.total_entries == 0


if __name__ == "__main__":
    pytest.main([__file__])
