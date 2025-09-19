from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, Field

from .intent_cache_types import (
    IntentCacheConfig,
    HandlerCacheConfig,
    CacheStrategy,
    CacheScope,
    DictData,
)


class MemoryConfig(BaseModel):
    """Configuration for the memory system including intent caching."""

    # Intent caching configuration
    intent_cache: IntentCacheConfig = Field(default_factory=IntentCacheConfig)

    # Memory engine configuration
    engine_config: Dict[str, Any] = Field(default_factory=dict)

    # Handler-specific cache configurations
    handler_cache_configs: Dict[str, HandlerCacheConfig] = Field(default_factory=dict)

    # Global settings
    enable_intent_caching: bool = True
    enable_fast_track: bool = True
    enable_semantic_matching: bool = True

    def get_handler_cache_config(self, handler_name: str) -> HandlerCacheConfig:
        """Get cache configuration for a specific handler."""
        if handler_name in self.handler_cache_configs:
            return self.handler_cache_configs[handler_name]

        # Return default config
        return HandlerCacheConfig(
            handler_name=handler_name,
            cache_data_type=DictData,
            strategy=CacheStrategy.SEMANTIC,
            scope=CacheScope.USER,
            ttl_seconds=3600,
            max_entries=1000,
            similarity_threshold=0.7,
            enable_fast_track=True,
        )

    def update_handler_cache_config(
        self, handler_name: str, config: HandlerCacheConfig
    ) -> None:
        """Update cache configuration for a specific handler."""
        self.handler_cache_configs[handler_name] = config
        self.intent_cache.handler_configs[handler_name] = config

    def disable_caching_for_handler(self, handler_name: str) -> None:
        """Disable caching for a specific handler."""
        config = self.get_handler_cache_config(handler_name)
        config.enable_fast_track = False
        config.ttl_seconds = 0  # Immediate expiration
        self.update_handler_cache_config(handler_name, config)

    def enable_caching_for_handler(
        self,
        handler_name: str,
        strategy: CacheStrategy = CacheStrategy.SEMANTIC,
        ttl_seconds: int = 3600,
    ) -> None:
        """Enable caching for a specific handler with custom settings."""
        config = HandlerCacheConfig(
            handler_name=handler_name,
            cache_data_type=DictData,
            strategy=strategy,
            scope=CacheScope.USER,
            ttl_seconds=ttl_seconds,
            max_entries=1000,
            similarity_threshold=0.7,
            enable_fast_track=True,
        )
        self.update_handler_cache_config(handler_name, config)


def create_default_memory_config() -> MemoryConfig:
    """Create a default memory configuration with intent caching enabled."""
    return MemoryConfig(
        intent_cache=IntentCacheConfig(
            enabled=True,
            default_strategy=CacheStrategy.SEMANTIC,
            default_ttl_seconds=3600,
            max_global_entries=10000,
            similarity_threshold=0.7,
            cleanup_interval_seconds=300,
        ),
        enable_intent_caching=True,
        enable_fast_track=True,
        enable_semantic_matching=True,
    )


def create_memory_config_from_dict(config_dict: Dict[str, Any]) -> MemoryConfig:
    """Create memory configuration from a dictionary."""
    # Extract intent cache config
    intent_cache_config = config_dict.get("intent_cache", {})
    intent_cache = IntentCacheConfig(**intent_cache_config)

    # Extract handler configs
    handler_configs = {}
    for handler_name, handler_config_dict in config_dict.get(
        "handler_cache_configs", {}
    ).items():
        handler_configs[handler_name] = HandlerCacheConfig(**handler_config_dict)

    return MemoryConfig(
        intent_cache=intent_cache,
        engine_config=config_dict.get("engine_config", {}),
        handler_cache_configs=handler_configs,
        enable_intent_caching=config_dict.get("enable_intent_caching", True),
        enable_fast_track=config_dict.get("enable_fast_track", True),
        enable_semantic_matching=config_dict.get("enable_semantic_matching", True),
    )
