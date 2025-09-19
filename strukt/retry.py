"""Retry functionality for LLM calls."""

from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Optional, Type
import random

from .logging import get_logger


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            Exception,  # Default to retry all exceptions
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if attempt <= 0:
            return 0.0

        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.0, delay)

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception should trigger a retry."""
        if attempt >= self.max_retries:
            return False

        return isinstance(exception, self.retryable_exceptions)


def retry_llm_call(
    config: Optional[RetryConfig] = None, operation_name: str = "llm_call"
) -> Callable:
    """
    Decorator to add retry functionality to LLM calls.

    Args:
        config: Retry configuration. If None, uses default config.
        operation_name: Name of the operation for logging.

    Returns:
        Decorated function with retry logic.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger("llm_retry")
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(
                            f"Retrying {operation_name} (attempt {attempt + 1}/{config.max_retries + 1})"
                        )

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            f"{operation_name} succeeded on attempt {attempt + 1}"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e, attempt):
                        logger.error(f"{operation_name} failed permanently: {e}")
                        raise e

                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warn(
                            f"{operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{operation_name} failed after {config.max_retries + 1} attempts: {e}"
                        )
                        raise e

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


def async_retry_llm_call(
    config: Optional[RetryConfig] = None, operation_name: str = "llm_call"
) -> Callable:
    """
    Decorator to add retry functionality to async LLM calls.

    Args:
        config: Retry configuration. If None, uses default config.
        operation_name: Name of the operation for logging.

    Returns:
        Decorated async function with retry logic.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            logger = get_logger("llm_retry")
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(
                            f"Retrying {operation_name} (attempt {attempt + 1}/{config.max_retries + 1})"
                        )

                    result = await func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            f"{operation_name} succeeded on attempt {attempt + 1}"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    if not config.should_retry(e, attempt):
                        logger.error(f"{operation_name} failed permanently: {e}")
                        raise e

                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warn(
                            f"{operation_name} failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"{operation_name} failed after {config.max_retries + 1} attempts: {e}"
                        )
                        raise e

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


class RetryableLLMClient:
    """
    Wrapper that adds retry functionality to any LLM client.

    This can be used to wrap existing LLM clients with retry logic.
    """

    def __init__(self, base_client: Any, retry_config: Optional[RetryConfig] = None):
        self._base = base_client
        self._retry_config = retry_config or RetryConfig()
        self._logger = get_logger("retryable_llm_client")

    def invoke(self, prompt: str, **kwargs: Any) -> Any:
        """Invoke with retry logic."""

        @retry_llm_call(self._retry_config, "invoke")
        def _invoke():
            return self._base.invoke(prompt, **kwargs)

        return _invoke()

    def structured(self, prompt: str, output_model: Type[Any], **kwargs: Any) -> Any:
        """Structured call with retry logic."""

        @retry_llm_call(self._retry_config, "structured")
        def _structured():
            return self._base.structured(prompt, output_model, **kwargs)

        return _structured()

    async def ainvoke(self, prompt: str, **kwargs: Any) -> Any:
        """Async invoke with retry logic."""

        @async_retry_llm_call(self._retry_config, "ainvoke")
        async def _ainvoke():
            return await self._base.ainvoke(prompt, **kwargs)

        return await _ainvoke()

    async def astructured(
        self, prompt: str, output_model: Type[Any], **kwargs: Any
    ) -> Any:
        """Async structured call with retry logic."""

        @async_retry_llm_call(self._retry_config, "astructured")
        async def _astructured():
            return await self._base.astructured(prompt, output_model, **kwargs)

        return await _astructured()

    # Delegate all other attributes to the base client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)
