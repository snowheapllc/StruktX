from __future__ import annotations

import re
from collections import defaultdict
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import hashlib

from .intent_cache_types import (
    CacheMatch,
    CacheStats,
    CacheStrategy,
    HandlerCacheConfig,
    IntentCacheConfig,
    IntentCacheEntry,
    IntentCacheEngine,
)
from ..logging import get_logger


class InMemoryIntentCacheEngine(IntentCacheEngine):
    """In-memory implementation of intent cache with semantic matching."""

    def __init__(self, config: IntentCacheConfig) -> None:
        self.config = config
        self._entries: Dict[str, IntentCacheEntry[Any]] = {}
        self._handler_entries: Dict[str, Set[str]] = defaultdict(set)
        self._user_entries: Dict[str, Set[str]] = defaultdict(set)
        self._unit_entries: Dict[str, Set[str]] = defaultdict(set)
        self._session_entries: Dict[str, Set[str]] = defaultdict(set)
        self._stats = CacheStats()
        self._last_cleanup = datetime.now(timezone.utc)
        self._logger = get_logger("intent_cache")

    def _get_handler_config(self, handler_name: str) -> HandlerCacheConfig:
        """Get handler-specific config or create default."""
        if handler_name in self.config.handler_configs:
            return self.config.handler_configs[handler_name]

        from .intent_cache_types import DictData

        return HandlerCacheConfig(
            handler_name=handler_name,
            cache_data_type=DictData,  # Default to DictData
            strategy=self.config.default_strategy,
            ttl_seconds=self.config.default_ttl_seconds,
            similarity_threshold=self.config.similarity_threshold,
        )

    def _calculate_similarity(
        self, text1: str, text2: str, strategy: CacheStrategy
    ) -> float:
        """Calculate similarity between two texts based on strategy."""
        if strategy == CacheStrategy.EXACT:
            return 1.0 if text1 == text2 else 0.0

        if strategy == CacheStrategy.SEMANTIC:
            return self._semantic_similarity(text1, text2)

        if strategy == CacheStrategy.FUZZY:
            return self._fuzzy_similarity(text1, text2)

        if strategy == CacheStrategy.HYBRID:
            semantic = self._semantic_similarity(text1, text2)
            fuzzy = self._fuzzy_similarity(text1, text2)
            return (semantic + fuzzy) / 2

        return 0.0

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using word overlap and TF-IDF-like scoring."""
        # Normalize texts
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        jaccard = intersection / union if union > 0 else 0.0

        # Weight by word frequency (simple TF-like scoring)
        common_words = words1.intersection(words2)
        if common_words:
            # Boost score for important words (longer words, less common patterns)
            word_weights = sum(len(word) for word in common_words) / len(common_words)
            weight_factor = min(1.2, 1.0 + (word_weights - 4) * 0.1)
            jaccard *= weight_factor

        return min(1.0, jaccard)

    def _fuzzy_similarity(self, text1: str, text2: str) -> float:
        """Calculate fuzzy string similarity using Levenshtein distance."""
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)

        if text1 == text2:
            return 1.0

        distance = self._levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove punctuation (optional - could be configurable)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def _get_candidate_entries(
        self,
        handler_name: str,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[IntentCacheEntry[Any]]:
        """Get candidate entries based on scope."""
        candidates = []

        # Start with handler-specific entries
        handler_entry_ids = self._handler_entries.get(handler_name, set())

        for entry_id in handler_entry_ids:
            if entry_id not in self._entries:
                continue

            entry = self._entries[entry_id]

            # Check scope
            if user_id and entry.user_id != user_id:
                continue
            if unit_id and entry.unit_id != unit_id:
                continue
            if session_id and entry.session_id != session_id:
                continue

            # Skip expired entries
            if entry.is_expired():
                continue

            candidates.append(entry)

        return candidates

    def _create_cache_key(self, text: str) -> str:
        """Create a normalized cache key from text."""
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(
        self,
        handler_name: str,
        cache_key: str,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[CacheMatch]:
        """Get a cached entry with semantic matching."""
        if not self.config.enabled:
            return None

        handler_config = self._get_handler_config(handler_name)
        candidates = self._get_candidate_entries(
            handler_name, user_id, unit_id, session_id
        )

        if not candidates:
            self._stats.misses += 1
            return None

        best_match = None
        best_score = 0.0

        for entry in candidates:
            # For semantic matching, compare original text with original text
            # For exact matching, compare hashed keys
            if handler_config.strategy == CacheStrategy.EXACT:
                similarity = self._calculate_similarity(
                    self._create_cache_key(cache_key),
                    entry.cache_key,
                    handler_config.strategy,
                )
            else:
                similarity = self._calculate_similarity(
                    cache_key, entry.original_text, handler_config.strategy
                )

            if similarity >= handler_config.similarity_threshold:
                if similarity > best_score:
                    best_score = similarity
                    best_match = CacheMatch(
                        entry=entry,
                        similarity_score=similarity,
                        match_type=handler_config.strategy,
                        is_exact=(similarity == 1.0),
                    )

        if best_match:
            best_match.entry.touch()
            self._stats.hits += 1
            self._stats.average_similarity = (
                self._stats.average_similarity * (self._stats.hits - 1) + best_score
            ) / self._stats.hits
            # Log cache hit with JSON format
            self._logger.cache_result(
                title="Cache Hit Result",
                data={
                    "cache_hit": True,
                    "handler_name": handler_name,
                    "similarity": best_score,
                    "match_type": handler_config.strategy.value,
                    "key": cache_key[:30] + "..." if len(cache_key) > 30 else cache_key,
                },
            )
        else:
            self._stats.misses += 1
            # Log cache miss
            self._logger.cache_miss(
                handler_name=handler_name,
                cache_key=cache_key,
                reason=f"No match found (threshold: {handler_config.similarity_threshold})",
            )

        return best_match

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
        if not self.config.enabled:
            return ""

        handler_config = self._get_handler_config(handler_name)

        # Create normalized cache key
        normalized_key = self._create_cache_key(cache_key)

        # Convert data to proper type if needed
        if isinstance(data, dict) and not isinstance(
            handler_config.cache_data_type, dict
        ):
            from .intent_cache_types import DictData

            if handler_config.cache_data_type == DictData:
                cached_data = DictData(data=data)
            else:
                cached_data = handler_config.cache_data_type(**data)
        else:
            cached_data = data

        # Create entry
        entry = IntentCacheEntry(
            handler_name=handler_name,
            cache_key=normalized_key,
            original_text=cache_key,  # Store original text for semantic matching
            cached_data=cached_data,
            user_id=user_id,
            unit_id=unit_id,
            session_id=session_id,
            ttl_seconds=ttl_seconds or handler_config.ttl_seconds,
        )

        # Store entry
        self._entries[entry.id] = entry

        # Update indexes
        self._handler_entries[handler_name].add(entry.id)
        if user_id:
            self._user_entries[user_id].add(entry.id)
        if unit_id:
            self._unit_entries[unit_id].add(entry.id)
        if session_id:
            self._session_entries[session_id].add(entry.id)

        # Enforce max entries limit
        self._enforce_limits(handler_name, handler_config)

        self._stats.total_entries = len(self._entries)

        # Log cache store with JSON format
        self._logger.cache_result(
            title="Cache Store Result",
            data={
                "cache_store": True,
                "handler_name": handler_name,
                "ttl_seconds": entry.ttl_seconds,
                "key": cache_key[:30] + "..." if len(cache_key) > 30 else cache_key,
            },
        )

        return entry.id

    def _enforce_limits(
        self, handler_name: str, handler_config: HandlerCacheConfig
    ) -> None:
        """Enforce entry limits for a handler."""
        handler_entries = self._handler_entries[handler_name]

        if len(handler_entries) > handler_config.max_entries:
            # Remove oldest entries (by last_accessed)
            entries_with_time = [
                (entry_id, self._entries[entry_id].last_accessed)
                for entry_id in handler_entries
                if entry_id in self._entries
            ]
            entries_with_time.sort(key=lambda x: x[1])

            # Remove excess entries
            to_remove = len(handler_entries) - handler_config.max_entries
            for entry_id, _ in entries_with_time[:to_remove]:
                self._remove_entry(entry_id)
                self._stats.evictions += 1

    def _remove_entry(self, entry_id: str) -> None:
        """Remove an entry and update all indexes."""
        if entry_id not in self._entries:
            return

        entry = self._entries[entry_id]

        # Remove from indexes
        self._handler_entries[entry.handler_name].discard(entry_id)
        if entry.user_id:
            self._user_entries[entry.user_id].discard(entry_id)
        if entry.unit_id:
            self._unit_entries[entry.unit_id].discard(entry_id)
        if entry.session_id:
            self._session_entries[entry.session_id].discard(entry_id)

        # Remove from main storage
        del self._entries[entry_id]

    def invalidate(
        self,
        handler_name: str,
        cache_key: Optional[str] = None,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries."""
        removed_count = 0
        handler_entries = list(self._handler_entries.get(handler_name, set()))

        for entry_id in handler_entries:
            if entry_id not in self._entries:
                continue

            entry = self._entries[entry_id]

            # Check filters
            if cache_key and entry.cache_key != self._create_cache_key(cache_key):
                continue
            if user_id and entry.user_id != user_id:
                continue
            if unit_id and entry.unit_id != unit_id:
                continue
            if session_id and entry.session_id != session_id:
                continue

            self._remove_entry(entry_id)
            removed_count += 1

        self._stats.total_entries = len(self._entries)
        return removed_count

    def cleanup(self) -> CacheStats:
        """Clean up expired entries and return statistics."""
        current_time = datetime.now(timezone.utc)

        # Only cleanup if enough time has passed
        if (
            current_time - self._last_cleanup
        ).total_seconds() < self.config.cleanup_interval_seconds:
            return self._stats

        expired_count = 0
        entries_to_remove = []

        for entry_id, entry in self._entries.items():
            if entry.is_expired():
                entries_to_remove.append(entry_id)
                expired_count += 1

        for entry_id in entries_to_remove:
            self._remove_entry(entry_id)

        self._stats.expired_entries += expired_count
        self._stats.total_entries = len(self._entries)
        self._last_cleanup = current_time

        return self._stats

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        # Calculate hit rate
        total_requests = self._stats.hits + self._stats.misses
        self._stats.hit_rate = (
            self._stats.hits / total_requests if total_requests > 0 else 0.0
        )

        # Update handler-specific stats
        for handler_name in self._handler_entries:
            handler_entries = self._handler_entries[handler_name]
            self._stats.handler_stats[handler_name] = {
                "entry_count": len(handler_entries),
                "active_entries": len(
                    [
                        eid
                        for eid in handler_entries
                        if eid in self._entries and not self._entries[eid].is_expired()
                    ]
                ),
            }

        return self._stats

    def log_stats(self) -> None:
        """Log cache statistics with pretty formatting."""
        stats = self.get_stats()
        stats_dict = {
            "total_entries": stats.total_entries,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
            "average_similarity": stats.average_similarity,
            "evictions": stats.evictions,
            "expired_entries": stats.expired_entries,
        }
        self._logger.cache_stats(stats_dict)
