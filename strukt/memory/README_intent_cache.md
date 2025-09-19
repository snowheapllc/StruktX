# Intent Caching System

The intent caching system provides intelligent caching capabilities for handlers in the StruktX framework. It supports semantic matching, fast-track caching, and configurable caching strategies to improve performance and reduce LLM usage.

## Features

- **Semantic Matching**: Cache entries are matched based on semantic similarity, not exact text matches
- **Multiple Strategies**: Support for exact, semantic, fuzzy, and hybrid matching strategies
- **Scoped Caching**: Cache entries can be scoped to users, units, sessions, or globally
- **Fast-Track Caching**: Immediate in-memory caching for frequently accessed data
- **TTL Support**: Time-to-live expiration for cache entries
- **Handler Integration**: Easy integration with existing handlers via mixins
- **Custom Data Models**: Support for Pydantic models as cached data
- **Statistics**: Comprehensive cache performance metrics

## Architecture

### Core Components

1. **IntentCacheEngine**: Abstract interface for cache implementations
2. **InMemoryIntentCacheEngine**: In-memory implementation with semantic matching
3. **HandlerCache**: Interface for handlers that want to implement caching
4. **HandlerCacheMixin**: Mixin classes for easy handler integration
5. **MemoryConfig**: Configuration management for the caching system

### Cache Strategies

- **EXACT**: Exact string matching (fastest, most restrictive)
- **SEMANTIC**: Semantic similarity using word overlap and TF-IDF-like scoring
- **FUZZY**: Fuzzy string matching using Levenshtein distance
- **HYBRID**: Combination of semantic and fuzzy matching

### Cache Scopes

- **GLOBAL**: Available to all users and units
- **USER**: Scoped to specific user
- **UNIT**: Scoped to specific unit
- **SESSION**: Scoped to current session

## Usage

### Basic Setup

```python
from strukt.memory import (
    IntentCacheConfig, 
    InMemoryIntentCacheEngine,
    MemoryConfig,
    create_default_memory_config
)

# Create configuration
config = create_default_memory_config()

# Create cache engine
cache_engine = InMemoryIntentCacheEngine(config.intent_cache)
```

### Handler Integration

#### Option 1: Using HandlerCacheMixin

```python
from strukt.memory import HandlerCacheMixin
from strukt.types import HandlerResult, InvocationState

class MyHandler(HandlerCacheMixin):
    def get_cache_config(self):
        return HandlerCacheConfig(
            handler_name="MyHandler",
            cache_data_type=dict,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            similarity_threshold=0.7
        )
    
    def should_cache(self, state: InvocationState, result: HandlerResult) -> bool:
        return result.status == "success"
    
    def build_cache_key(self, state: InvocationState) -> str:
        return f"my_handler:{state.text}"
    
    def extract_cache_data(self, state: InvocationState, result: HandlerResult):
        return {
            "response": result.response,
            "status": result.status,
            "query": state.text
        }
    
    def handle(self, state: InvocationState, parts: list) -> HandlerResult:
        # Your handler logic here
        return HandlerResult(response="Processed", status="success")
```

#### Option 2: Using CacheAwareHandler

```python
from strukt.memory import CacheAwareHandler

class MyCachedHandler(CacheAwareHandler):
    def get_cache_config(self):
        return HandlerCacheConfig(
            handler_name="MyCachedHandler",
            cache_data_type=dict,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600
        )
    
    def handle(self, state: InvocationState, parts: list) -> HandlerResult:
        # Your handler logic here
        return HandlerResult(response="Processed", status="success")

# Usage with caching
result = handler.handle_with_cache(state, cache_engine)
```

### Custom Data Models

```python
from pydantic import BaseModel
from strukt.memory import SemanticCacheHandlerMixin

class WeatherData(BaseModel):
    location: str
    temperature: float
    condition: str
    timestamp: datetime

class WeatherHandler(SemanticCacheHandlerMixin):
    def __init__(self):
        super().__init__()
        self.set_cache_data_type(WeatherData)
    
    def extract_cache_data(self, state: InvocationState, result: HandlerResult) -> WeatherData:
        return WeatherData(
            location="New York",
            temperature=22.5,
            condition="sunny",
            timestamp=datetime.now()
        )
```

### Fast-Track Caching

```python
from strukt.memory import FastTrackHandlerMixin

class FastHandler(FastTrackHandlerMixin):
    def handle(self, state: InvocationState, parts: list) -> HandlerResult:
        # Check fast track first
        cached = self.get_fast_track(state)
        if cached:
            return cached
        
        # Process normally
        result = self.process_request(state)
        
        # Cache for next time
        self.set_fast_track(state, result)
        return result
```

## Configuration

### Global Configuration

```python
config = MemoryConfig(
    enable_intent_caching=True,
    enable_fast_track=True,
    enable_semantic_matching=True,
    intent_cache=IntentCacheConfig(
        enabled=True,
        default_strategy=CacheStrategy.SEMANTIC,
        default_ttl_seconds=3600,
        max_global_entries=10000,
        similarity_threshold=0.7,
        cleanup_interval_seconds=300
    )
)
```

### Handler-Specific Configuration

```python
# Configure specific handler
config.update_handler_cache_config(
    "WeatherHandler",
    HandlerCacheConfig(
        handler_name="WeatherHandler",
        cache_data_type=WeatherData,
        strategy=CacheStrategy.SEMANTIC,
        scope=CacheScope.USER,
        ttl_seconds=1800,  # 30 minutes
        max_entries=500,
        similarity_threshold=0.8,
        enable_fast_track=True
    )
)

# Disable caching for a handler
config.disable_caching_for_handler("NoCacheHandler")
```

## Cache Operations

### Storing Data

```python
# Store data in cache
entry_id = cache_engine.put(
    handler_name="MyHandler",
    cache_key="user query",
    data={"response": "cached response"},
    user_id="user123",
    unit_id="unit456",
    ttl_seconds=3600
)
```

### Retrieving Data

```python
# Get cached data
match = cache_engine.get(
    handler_name="MyHandler",
    cache_key="similar query",
    user_id="user123",
    unit_id="unit456"
)

if match and match.is_valid:
    print(f"Cached response: {match.entry.cached_data}")
    print(f"Similarity score: {match.similarity_score}")
```

### Cache Management

```python
# Invalidate cache entries
invalidated = cache_engine.invalidate(
    handler_name="MyHandler",
    user_id="user123"
)

# Cleanup expired entries
stats = cache_engine.cleanup()

# Get cache statistics
stats = cache_engine.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Total entries: {stats.total_entries}")
```

## Performance Considerations

### Similarity Thresholds

- **0.9-1.0**: Very strict matching (exact or near-exact)
- **0.7-0.9**: Good balance for most use cases
- **0.5-0.7**: More permissive matching
- **<0.5**: Very permissive (may match unrelated queries)

### Cache Sizing

- **Small handlers**: 100-500 entries
- **Medium handlers**: 500-2000 entries
- **Large handlers**: 2000-10000 entries
- **Global cache**: 10000+ entries

### TTL Settings

- **Real-time data**: 60-300 seconds
- **Semi-static data**: 1800-3600 seconds (30-60 minutes)
- **Static data**: 86400+ seconds (24+ hours)

## Best Practices

1. **Choose Appropriate Strategies**: Use exact matching for precise queries, semantic for natural language
2. **Set Reasonable TTLs**: Balance freshness with performance
3. **Monitor Cache Performance**: Use statistics to optimize settings
4. **Use Scoped Caching**: Scope cache entries appropriately for your use case
5. **Implement Fast-Track**: Use fast-track caching for frequently accessed data
6. **Clean Up Regularly**: Set up periodic cleanup to remove expired entries

## Examples

See `examples/intent_cache_example.py` for comprehensive examples demonstrating:
- Basic caching setup
- Different caching strategies
- Handler integration
- Custom data models
- Performance optimization

## Testing

Run the test suite to verify functionality:

```bash
pytest tests/test_intent_cache.py -v
```

The tests cover:
- Basic cache operations
- Semantic matching
- User scoping
- TTL expiration
- Cache cleanup
- Statistics
- Handler integration
