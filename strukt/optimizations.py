"""Performance optimization utilities for StruktX framework.

This module provides:
- Circuit Breaker pattern for fault tolerance
- Performance monitoring and metrics
- Streaming response handlers
- Multi-level caching
- Rate limiting with semaphores
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, AsyncIterator, Callable, Generic, TypeVar

from .logging import get_logger

T = TypeVar("T")


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2  # Successes needed in HALF_OPEN to close
    timeout: float = 10.0  # Operation timeout


class CircuitBreaker:
    """Circuit breaker for fault tolerance.
    Prevents cascading failures by opening circuit after threshold failures.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self._logger = get_logger("circuit_breaker")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.state != CircuitState.OPEN or self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    async def call(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            self._logger.info("Circuit breaker entering HALF_OPEN state")

        # Reject if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            # Execute operation with timeout
            result = await asyncio.wait_for(
                operation(*args, **kwargs)
                if asyncio.iscoroutinefunction(operation)
                else asyncio.to_thread(operation, *args, **kwargs),
                timeout=self.config.timeout,
            )

            # Record success
            self._on_success()
            return result

        except Exception as e:
            # Record failure
            self._on_failure(e)
            raise

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self._logger.info("Circuit breaker CLOSED after recovery")
        else:
            self.failure_count = 0

    def _on_failure(self, error: Exception):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._logger.warn(
                f"Circuit breaker OPEN after failure in HALF_OPEN: {error}"
            )
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._logger.warn(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )


def circuit_breaker(config: CircuitBreakerConfig | None = None):
    """Decorator to add circuit breaker to async functions."""
    breaker = CircuitBreaker(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Performance Monitoring
# ============================================================================


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""

    name: str
    count: int = 0
    total_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    error_count: int = 0
    last_error: str | None = None

    @property
    def avg_duration(self) -> float:
        """Average duration in seconds."""
        return self.total_duration / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        return (
            ((self.count - self.error_count) / self.count * 100)
            if self.count > 0
            else 100.0
        )


class PerformanceMonitor:
    """Real-time performance monitoring and metrics collection."""

    def __init__(self, window_size: int = 1000):
        self.metrics: dict[str, OperationMetrics] = defaultdict(
            lambda: OperationMetrics(name="unknown")
        )
        self.recent_latencies: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._logger = get_logger("performance_monitor")

    async def track(
        self,
        operation_name: str,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Track operation execution with metrics."""
        start_time = time.perf_counter()
        metric = self.metrics[operation_name]
        metric.name = operation_name
        metric.count += 1

        try:
            result = (
                await operation(*args, **kwargs)
                if asyncio.iscoroutinefunction(operation)
                else await asyncio.to_thread(operation, *args, **kwargs)
            )

            # Record timing
            duration = time.perf_counter() - start_time
            metric.total_duration += duration
            metric.min_duration = min(metric.min_duration, duration)
            metric.max_duration = max(metric.max_duration, duration)
            self.recent_latencies[operation_name].append(duration)

            return result

        except Exception as e:
            metric.error_count += 1
            metric.last_error = str(e)
            duration = time.perf_counter() - start_time
            metric.total_duration += duration
            raise

    def get_metrics(
        self, operation_name: str | None = None
    ) -> dict[str, OperationMetrics] | OperationMetrics:
        """Get metrics for specific operation or all operations."""
        if operation_name:
            return self.metrics.get(
                operation_name, OperationMetrics(name=operation_name)
            )
        return dict(self.metrics)

    def get_p95_latency(self, operation_name: str) -> float:
        """Get 95th percentile latency for operation."""
        latencies = list(self.recent_latencies.get(operation_name, []))
        if not latencies:
            return 0.0
        latencies.sort()
        idx = int(len(latencies) * 0.95)
        return latencies[idx] if idx < len(latencies) else latencies[-1]

    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.recent_latencies.clear()


def monitor_performance(monitor: PerformanceMonitor, operation_name: str):
    """Decorator to add performance monitoring to async functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await monitor.track(operation_name, func, *args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Multi-Level Cache
# ============================================================================


class CacheLevel(Enum):
    """Cache level enumeration."""

    L1_MEMORY = 1
    L2_DISTRIBUTED = 2


class HierarchicalCache(Generic[T]):
    """Multi-level caching with L1 (memory) and L2 (distributed) tiers."""

    def __init__(self, l1_maxsize: int = 1000, l2_client: Any | None = None):
        self.l1_cache: dict[str, T] = {}
        self.l1_maxsize = l1_maxsize
        self.l2_client = l2_client  # Redis or similar
        self._access_count: dict[str, int] = defaultdict(int)
        self._logger = get_logger("hierarchical_cache")

    async def get_or_compute(
        self, key: str, compute_fn: Callable[[], Any], ttl: int = 3600
    ) -> T:
        """Get from cache or compute if missing."""
        # Try L1 cache
        if key in self.l1_cache:
            self._access_count[key] += 1
            return self.l1_cache[key]

        # Try L2 cache if available
        if self.l2_client:
            try:
                l2_value = await self._get_from_l2(key)
                if l2_value is not None:
                    # Promote to L1
                    await self._set_l1(key, l2_value)
                    return l2_value
            except Exception as e:
                self._logger.warn(f"L2 cache lookup failed: {e}")

        # Compute value
        value = (
            await compute_fn()
            if asyncio.iscoroutinefunction(compute_fn)
            else await asyncio.to_thread(compute_fn)
        )

        # Store in both caches
        await self._set_l1(key, value)
        if self.l2_client:
            await self._set_l2(key, value, ttl)

        return value

    async def _set_l1(self, key: str, value: T):
        """Set L1 cache with LRU eviction."""
        if len(self.l1_cache) >= self.l1_maxsize:
            # Evict least accessed
            lru_key = min(self._access_count, key=self._access_count.get)
            del self.l1_cache[lru_key]
            del self._access_count[lru_key]

        self.l1_cache[key] = value
        self._access_count[key] = 1

    async def _get_from_l2(self, key: str) -> T | None:
        """Get from L2 cache (distributed)."""
        if not self.l2_client:
            return None
        # Implement based on your distributed cache (Redis, etc.)
        return None

    async def _set_l2(self, key: str, value: T, ttl: int):
        """Set L2 cache with TTL."""
        if not self.l2_client:
            return
        # Implement based on your distributed cache (Redis, etc.)
        await asyncio.sleep(0)  # Placeholder for async operation

    def clear(self):
        """Clear L1 cache."""
        self.l1_cache.clear()
        self._access_count.clear()


# ============================================================================
# Adaptive Load Balancing
# ============================================================================


@dataclass
class HandlerScore:
    """Performance score for a handler."""

    handler_id: str
    avg_latency: float = 0.0
    success_rate: float = 100.0
    active_requests: int = 0

    @property
    def score(self) -> float:
        """Calculate overall score (lower is better)."""
        # Penalize high latency, low success rate, and high load
        latency_penalty = self.avg_latency * 1000  # Convert to ms
        success_penalty = (100 - self.success_rate) * 10
        load_penalty = self.active_requests * 100
        return latency_penalty + success_penalty + load_penalty


class AdaptiveLoadBalancer:
    """Route requests to best-performing handlers."""

    def __init__(self):
        self.handler_scores: dict[str, HandlerScore] = {}
        self.handler_metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._logger = get_logger("load_balancer")

    def select_handler(self, handler_ids: list[str]) -> str:
        """Select optimal handler based on metrics."""
        if not handler_ids:
            raise ValueError("No handlers available")

        if len(handler_ids) == 1:
            return handler_ids[0]

        # Get scores for all handlers
        scores = []
        for handler_id in handler_ids:
            if handler_id not in self.handler_scores:
                self.handler_scores[handler_id] = HandlerScore(handler_id=handler_id)
            scores.append((handler_id, self.handler_scores[handler_id].score))

        # Select handler with lowest score
        selected = min(scores, key=lambda x: x[1])[0]
        self.handler_scores[selected].active_requests += 1
        return selected

    def record_result(self, handler_id: str, duration: float, success: bool):
        """Record handler execution result."""
        if handler_id not in self.handler_scores:
            self.handler_scores[handler_id] = HandlerScore(handler_id=handler_id)

        score = self.handler_scores[handler_id]
        score.active_requests = max(0, score.active_requests - 1)

        # Update metrics
        metrics = self.handler_metrics[handler_id]
        metrics.append({"duration": duration, "success": success})

        # Recalculate scores
        if metrics:
            total_duration = sum(m["duration"] for m in metrics)
            success_count = sum(1 for m in metrics if m["success"])
            score.avg_latency = total_duration / len(metrics)
            score.success_rate = (success_count / len(metrics)) * 100


# ============================================================================
# Rate Limiting
# ============================================================================


class RateLimiter:
    """Semaphore-based rate limiting for concurrent operations."""

    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self._active_count = 0
        self._logger = get_logger("rate_limiter")

    async def execute(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute operation with rate limiting."""
        async with self.semaphore:
            self._active_count += 1
            try:
                return (
                    await operation(*args, **kwargs)
                    if asyncio.iscoroutinefunction(operation)
                    else await asyncio.to_thread(operation, *args, **kwargs)
                )
            finally:
                self._active_count -= 1

    @property
    def active_requests(self) -> int:
        """Number of currently active requests."""
        return self._active_count

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active_count


# ============================================================================
# Streaming Response Handler
# ============================================================================


class StreamBuffer:
    """Buffer for streaming responses."""

    def __init__(self, chunk_size: int = 1024):
        self.buffer: list[str] = []
        self.chunk_size = chunk_size

    def write(self, data: str):
        """Write data to buffer."""
        self.buffer.append(data)

    def flush(self) -> str:
        """Flush buffer and return contents."""
        result = "".join(self.buffer)
        self.buffer.clear()
        return result

    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self.buffer)


async def stream_tokens(response_iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Stream tokens from an async iterator with buffering."""
    buffer = StreamBuffer()

    async for token in response_iterator:
        buffer.write(token)
        if len(buffer) >= buffer.chunk_size:
            yield buffer.flush()

    # Flush remaining
    remaining = buffer.flush()
    if remaining:
        yield remaining


# ============================================================================
# Batch Processing
# ============================================================================


class BatchProcessor(Generic[T]):
    """Batch processor for grouping operations."""

    def __init__(self, batch_size: int = 16, max_wait: float = 0.1):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue: asyncio.Queue = asyncio.Queue()
        self._logger = get_logger("batch_processor")

    async def add(self, item: T) -> Any:
        """Add item to batch queue."""
        future = asyncio.Future()
        await self.queue.put((item, future))
        return await future

    async def process_batches(self, process_fn: Callable[[list[T]], list[Any]]):
        """Process items in batches."""
        while True:
            batch: list[tuple[T, asyncio.Future]] = []
            deadline = time.time() + self.max_wait

            # Collect batch
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(), timeout=deadline - time.time()
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                await asyncio.sleep(0.01)
                continue

            # Process batch
            try:
                items = [item for item, _ in batch]
                results = (
                    await process_fn(items)
                    if asyncio.iscoroutinefunction(process_fn)
                    else await asyncio.to_thread(process_fn, items)
                )

                # Set results
                for (_, future), result in zip(batch, results):
                    future.set_result(result)
            except Exception as e:
                # Set exception for all futures
                for _, future in batch:
                    future.set_exception(e)
