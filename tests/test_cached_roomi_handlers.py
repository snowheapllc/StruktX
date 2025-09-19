"""
Test cached Roomi handlers to ensure intent caching works correctly.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from strukt.types import InvocationState, HandlerResult
from strukt.memory import InMemoryIntentCacheEngine, create_default_memory_config

from strukt.extensions.roomi.cached_handlers import (
    CachedWeatherHandler,
    CachedFutureEventHandler,
    CachedHelpdeskHandler,
    CachedEventHandler,
    CachedBillHandler,
    CachedNotificationHandler,
    CachedAmenityHandler,
)


class TestCachedRoomiHandlers:
    """Test cached Roomi handlers functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create cache engine
        memory_config = create_default_memory_config()
        self.cache_engine = InMemoryIntentCacheEngine(memory_config.intent_cache)
        
        # Create mock toolkit and LLM client
        self.mock_toolkit = Mock()
        self.mock_llm = Mock()
    
    def test_cached_weather_handler(self):
        """Test cached weather handler."""
        # Create handler
        handler = CachedWeatherHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        # Mock the toolkit response
        self.mock_toolkit.get_current_weather_data = AsyncMock(return_value={
            "success": True,
            "message": "Weather data retrieved",
            "weather_data": {
                "location": "Dubai",
                "temperature": 25.0,
                "condition": "sunny"
            }
        })
        
        # First request - should miss cache
        state1 = InvocationState(
            text="What's the weather like in Dubai?",
            parts=["What's", "the", "weather", "like", "in", "Dubai?"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result1 = handler.handle_with_cache(state1, self.cache_engine)
        assert result1.status == "WEATHER_SUCCESS"
        assert "Dubai" in str(result1.response)
        
        # Second similar request - should hit cache
        state2 = InvocationState(
            text="How's the weather in Dubai?",
            parts=["How's", "the", "weather", "in", "Dubai?"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result2 = handler.handle_with_cache(state2, self.cache_engine)
        assert result2.status == "WEATHER_SUCCESS"
        assert "Dubai" in str(result2.response)
        
        # Check cache stats
        stats = self.cache_engine.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1
        assert stats.hit_rate > 0
    
    def test_cached_future_event_handler(self):
        """Test cached future event handler."""
        # Create handler
        handler = CachedFutureEventHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        # Mock the toolkit response
        self.mock_toolkit.create_event = AsyncMock(return_value={
            "success": True,
            "message": "Event created successfully",
            "event_data": {
                "event_id": "evt_123",
                "title": "Test Event",
                "date": "2024-01-01"
            }
        })
        
        # First request - should miss cache
        state1 = InvocationState(
            text="Schedule a meeting for tomorrow",
            parts=["Schedule", "a", "meeting", "for", "tomorrow"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result1 = handler.handle_with_cache(state1, self.cache_engine)
        assert result1.status in ["EVENT_SUCCESS", "EVENT_CREATED"]
        
        # Second similar request - should hit cache
        state2 = InvocationState(
            text="Create a meeting for tomorrow",
            parts=["Create", "a", "meeting", "for", "tomorrow"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result2 = handler.handle_with_cache(state2, self.cache_engine)
        assert result2.status in ["EVENT_SUCCESS", "EVENT_CREATED"]
        
        # Check cache stats
        stats = self.cache_engine.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1
    
    def test_cached_helpdesk_handler(self):
        """Test cached helpdesk handler."""
        # Create handler
        handler = CachedHelpdeskHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        # Mock the toolkit response
        self.mock_toolkit.create_ticket = AsyncMock(return_value={
            "success": True,
            "message": "Ticket created successfully",
            "ticket_data": {
                "ticket_id": "tkt_123",
                "title": "Test Issue",
                "status": "open"
            }
        })
        
        # First request - should miss cache
        state1 = InvocationState(
            text="I need help with my internet connection",
            parts=["I", "need", "help", "with", "my", "internet", "connection"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result1 = handler.handle_with_cache(state1, self.cache_engine)
        assert result1.status in ["HELPDESK_SUCCESS", "TICKET_CREATED"]
        
        # Second similar request - should hit cache
        state2 = InvocationState(
            text="Help me with internet issues",
            parts=["Help", "me", "with", "internet", "issues"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result2 = handler.handle_with_cache(state2, self.cache_engine)
        assert result2.status in ["HELPDESK_SUCCESS", "TICKET_CREATED"]
        
        # Check cache stats
        stats = self.cache_engine.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1
    
    def test_cache_key_generation(self):
        """Test cache key generation for different handlers."""
        # Create handlers
        weather_handler = CachedWeatherHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        event_handler = CachedFutureEventHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        # Test weather cache key
        state1 = InvocationState(
            text="What's the weather like in Dubai?",
            parts=["What's", "the", "weather", "like", "in", "Dubai?"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        weather_key = weather_handler.build_cache_key(state1)
        assert "weather:" in weather_key
        assert "dubai" in weather_key.lower()
        
        # Test event cache key
        state2 = InvocationState(
            text="Schedule a meeting for tomorrow",
            parts=["Schedule", "a", "meeting", "for", "tomorrow"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        event_key = event_handler.build_cache_key(state2)
        assert "future_event:" in event_key
    
    def test_cache_data_extraction(self):
        """Test cache data extraction and application."""
        # Create handler
        handler = CachedWeatherHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        # Test data extraction
        state = InvocationState(
            text="What's the weather like in Dubai?",
            parts=["What's", "the", "weather", "like", "in", "Dubai?"],
            context={"user_id": "user123", "unit_id": "unit456"}
        )
        
        result = HandlerResult(
            response={
                "status": "success",
                "message": "Weather data retrieved",
                "current_date": "2024-01-01",
                "weather_data": {
                    "location": "Dubai",
                    "temperature": 25.0,
                    "condition": "sunny"
                }
            },
            status="WEATHER_SUCCESS"
        )
        
        # Extract cache data
        cache_data = handler.extract_cache_data(state, result)
        assert cache_data.location == "Dubai"
        assert cache_data.status == "success"
        assert cache_data.weather_data["temperature"] == 25.0
        
        # Apply cached data
        cached_result = handler.apply_cached_data(state, cache_data)
        assert cached_result.status == "WEATHER_SUCCESS"
        assert "Dubai" in str(cached_result.response)
    
    def test_user_scoping(self):
        """Test that cache entries are scoped by user."""
        # Create handler
        handler = CachedWeatherHandler(
            toolkit=self.mock_toolkit,
            llm=self.mock_llm
        )
        
        # Mock the toolkit response
        self.mock_toolkit.get_current_weather_data = AsyncMock(return_value={
            "success": True,
            "message": "Weather data retrieved",
            "weather_data": {
                "location": "Dubai",
                "temperature": 25.0,
                "condition": "sunny"
            }
        })
        
        # Request from user1
        state1 = InvocationState(
            text="What's the weather like in Dubai?",
            parts=["What's", "the", "weather", "like", "in", "Dubai?"],
            context={"user_id": "user1", "unit_id": "unit1"}
        )
        
        result1 = handler.handle_with_cache(state1, self.cache_engine)
        assert result1.status == "WEATHER_SUCCESS"
        
        # Similar request from user2 - should miss cache due to user scoping
        state2 = InvocationState(
            text="How's the weather in Dubai?",
            parts=["How's", "the", "weather", "in", "Dubai?"],
            context={"user_id": "user2", "unit_id": "unit1"}
        )
        
        result2 = handler.handle_with_cache(state2, self.cache_engine)
        assert result2.status == "WEATHER_SUCCESS"
        
        # Check cache stats - should have 2 misses, 0 hits due to user scoping
        stats = self.cache_engine.get_stats()
        assert stats.misses >= 2
        assert stats.hits == 0  # No hits due to different users


if __name__ == "__main__":
    pytest.main([__file__])
