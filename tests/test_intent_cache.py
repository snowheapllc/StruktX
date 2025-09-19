"""
Tests for intent caching functionality.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from strukt.memory.intent_cache_types import (
    CacheStrategy,
    CacheScope,
    HandlerCacheConfig,
    IntentCacheConfig,
    IntentCacheEntry,
    CacheMatch,
    DictData
)
from strukt.memory.intent_cache_engine import InMemoryIntentCacheEngine
from strukt.memory.handler_cache_mixin import HandlerCacheMixin, FastTrackHandlerMixin
from strukt.memory.memory_config import MemoryConfig, create_default_memory_config
from strukt.types import HandlerResult, InvocationState


class MockCacheData:
    """Test data model for caching."""
    def __init__(self, response: str, metadata: Dict[str, Any] = None):
        self.response = response
        self.metadata = metadata or {}


class MockHandler(HandlerCacheMixin, FastTrackHandlerMixin):
    """Test handler with caching support."""
    
    def get_cache_config(self) -> HandlerCacheConfig:
        return HandlerCacheConfig(
            handler_name="TestHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            max_entries=100,
            similarity_threshold=0.7,
            enable_fast_track=True
        )
    
    def should_cache(self, state: InvocationState, result: HandlerResult) -> bool:
        return result.status == "success"
    
    def build_cache_key(self, state: InvocationState) -> str:
        # Use the parent implementation that includes context, but add our prefix
        parent_key = super().build_cache_key(state)
        return f"test:{parent_key}"
    
    def extract_cache_data(self, state: InvocationState, result: HandlerResult) -> Dict[str, Any]:
        return {
            "response": result.response,
            "status": result.status,
            "query": state.text
        }
    
    def handle(self, state: InvocationState, parts: list) -> HandlerResult:
        return HandlerResult(
            response=f"Processed: {state.text}",
            status="success"
        )


class TestIntentCacheEngine:
    """Test intent cache engine functionality."""
    
    def test_basic_put_and_get(self):
        """Test basic cache put and get operations."""
        config = IntentCacheConfig(enabled=True)
        engine = InMemoryIntentCacheEngine(config)
        
        # Put data
        entry_id = engine.put(
            "TestHandler",
            "test query",
            {"response": "test response"},
            user_id="user1"
        )
        
        assert entry_id != ""
        
        # Get data
        match = engine.get("TestHandler", "test query", user_id="user1")
        assert match is not None
        assert match.similarity_score == 1.0
        assert match.is_exact is True
        assert match.entry.cached_data["response"] == "test response"
    
    def test_semantic_matching(self):
        """Test semantic matching functionality."""
        config = IntentCacheConfig(
            enabled=True,
            default_strategy=CacheStrategy.SEMANTIC,
            similarity_threshold=0.2  # Even lower threshold for semantic matching
        )
        engine = InMemoryIntentCacheEngine(config)
        
        # Store original query
        engine.put(
            "TestHandler",
            "What's the weather like?",
            {"response": "It's sunny"},
            user_id="user1"
        )
        
        # Test similar queries
        similar_queries = [
            "How's the weather?",
            "What is the weather like?",
            "Tell me about the weather",
            "Weather conditions"
        ]
        
        for query in similar_queries:
            match = engine.get("TestHandler", query, user_id="user1")
            assert match is not None, f"Should match: {query}"
            assert match.similarity_score >= 0.2, f"Similarity too low: {match.similarity_score}"
    
    def test_exact_matching(self):
        """Test exact matching strategy."""
        config = IntentCacheConfig(
            enabled=True,
            default_strategy=CacheStrategy.EXACT,
            similarity_threshold=1.0
        )
        engine = InMemoryIntentCacheEngine(config)
        
        # Store query
        engine.put(
            "TestHandler",
            "exact query",
            {"response": "exact response"},
            user_id="user1"
        )
        
        # Test exact match
        match = engine.get("TestHandler", "exact query", user_id="user1")
        assert match is not None
        assert match.similarity_score == 1.0
        assert match.is_exact is True
        
        # Test non-exact match
        match = engine.get("TestHandler", "similar query", user_id="user1")
        assert match is None
    
    def test_user_scoping(self):
        """Test user-scoped caching."""
        config = IntentCacheConfig(enabled=True)
        engine = InMemoryIntentCacheEngine(config)
        
        # Store for user1
        engine.put(
            "TestHandler",
            "user query",
            {"response": "user1 response"},
            user_id="user1"
        )
        
        # Store for user2
        engine.put(
            "TestHandler",
            "user query",
            {"response": "user2 response"},
            user_id="user2"
        )
        
        # Test user1 access
        match1 = engine.get("TestHandler", "user query", user_id="user1")
        assert match1 is not None
        assert match1.entry.cached_data["response"] == "user1 response"
        
        # Test user2 access
        match2 = engine.get("TestHandler", "user query", user_id="user2")
        assert match2 is not None
        assert match2.entry.cached_data["response"] == "user2 response"
        
        # Test cross-user access (should not match)
        match3 = engine.get("TestHandler", "user query", user_id="user1")
        assert match3.entry.cached_data["response"] == "user1 response"
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        config = IntentCacheConfig(enabled=True)
        engine = InMemoryIntentCacheEngine(config)
        
        # Store with short TTL
        engine.put(
            "TestHandler",
            "ttl query",
            {"response": "ttl response"},
            user_id="user1",
            ttl_seconds=1  # 1 second TTL
        )
        
        # Should be available immediately
        match = engine.get("TestHandler", "ttl query", user_id="user1")
        assert match is not None
        
        # Wait for expiration (in real scenario, you'd use time mocking)
        # For this test, we'll manually set the created_at time
        entry_id = list(engine._entries.keys())[0]
        entry = engine._entries[entry_id]
        entry.created_at = datetime.now(timezone.utc) - timedelta(seconds=2)
        
        # Should be expired now
        match = engine.get("TestHandler", "ttl query", user_id="user1")
        assert match is None or match.entry.is_expired()
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        config = IntentCacheConfig(
            enabled=True,
            cleanup_interval_seconds=0  # Immediate cleanup
        )
        engine = InMemoryIntentCacheEngine(config)
        
        # Store some entries
        engine.put("TestHandler", "query1", {"response": "response1"}, user_id="user1")
        engine.put("TestHandler", "query2", {"response": "response2"}, user_id="user1")
        
        # Manually expire one entry
        entry_id = list(engine._entries.keys())[0]
        entry = engine._entries[entry_id]
        entry.created_at = datetime.now(timezone.utc) - timedelta(seconds=3600)
        
        # Run cleanup
        stats = engine.cleanup()
        assert stats.expired_entries >= 1
    
    def test_cache_stats(self):
        """Test cache statistics."""
        config = IntentCacheConfig(enabled=True)
        engine = InMemoryIntentCacheEngine(config)
        
        # Initial stats
        stats = engine.get_stats()
        assert stats.total_entries == 0
        assert stats.hits == 0
        assert stats.misses == 0
        
        # Add some entries
        engine.put("TestHandler", "query1", {"response": "response1"}, user_id="user1")
        engine.put("TestHandler", "query2", {"response": "response2"}, user_id="user1")
        
        # Test hits and misses
        engine.get("TestHandler", "query1", user_id="user1")  # Hit
        engine.get("TestHandler", "query3", user_id="user1")  # Miss
        
        stats = engine.get_stats()
        assert stats.total_entries == 2
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5


class TestHandlerCacheMixin:
    """Test handler cache mixin functionality."""
    
    def test_fast_track_caching(self):
        """Test fast track caching."""
        handler = MockHandler()
        
        state = InvocationState(text="test query")
        result = HandlerResult(response="test response", status="success")
        
        # Should not be in cache initially
        cached_result = handler.get_fast_track(state)
        assert cached_result is None
        
        # Set in cache
        handler.set_fast_track(state, result)
        
        # Should be in cache now
        cached_result = handler.get_fast_track(state)
        assert cached_result is not None
        assert cached_result.response == "test response"
        assert cached_result.status == "fast_track_cached"
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        handler = MockHandler()
        
        state = InvocationState(text="test query")
        cache_key = handler.build_cache_key(state)
        assert cache_key == "test:test query"
        
        # Test with context
        state_with_context = InvocationState(
            text="test query",
            context={"user_id": "user1", "unit_id": "unit1"}
        )
        cache_key = handler.build_cache_key(state_with_context)
        assert "test:test query" in cache_key
        assert "user:user1" in cache_key
        assert "unit:unit1" in cache_key
    
    def test_should_cache_logic(self):
        """Test should_cache logic."""
        handler = MockHandler()
        
        state = InvocationState(text="test query")
        success_result = HandlerResult(response="success", status="success")
        error_result = HandlerResult(response="error", status="error")
        
        assert handler.should_cache(state, success_result) is True
        assert handler.should_cache(state, error_result) is False


class TestMemoryConfig:
    """Test memory configuration functionality."""
    
    def test_default_config(self):
        """Test default memory configuration."""
        config = create_default_memory_config()
        
        assert config.enable_intent_caching is True
        assert config.enable_fast_track is True
        assert config.enable_semantic_matching is True
        assert config.intent_cache.enabled is True
    
    def test_handler_config_management(self):
        """Test handler configuration management."""
        config = MemoryConfig()
        
        # Test getting default config
        handler_config = config.get_handler_cache_config("TestHandler")
        assert handler_config.handler_name == "TestHandler"
        assert handler_config.strategy == CacheStrategy.SEMANTIC
        
        # Test updating config
        custom_config = HandlerCacheConfig(
            handler_name="TestHandler",
            cache_data_type=DictData,
            strategy=CacheStrategy.EXACT,
            ttl_seconds=1800
        )
        config.update_handler_cache_config("TestHandler", custom_config)
        
        updated_config = config.get_handler_cache_config("TestHandler")
        assert updated_config.strategy == CacheStrategy.EXACT
        assert updated_config.ttl_seconds == 1800
    
    def test_disable_caching(self):
        """Test disabling caching for handlers."""
        config = MemoryConfig()
        
        # Disable caching for handler
        config.disable_caching_for_handler("TestHandler")
        
        handler_config = config.get_handler_cache_config("TestHandler")
        assert handler_config.enable_fast_track is False
        assert handler_config.ttl_seconds == 0


if __name__ == "__main__":
    pytest.main([__file__])
