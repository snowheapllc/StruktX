from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces import MemoryEngine
from .memory import batch_add_edges, batch_add_nodes, build_edge, build_node
from .memory_types import KnowledgeCategory, KnowledgeEdge, KnowledgeNode


class KnowledgeStore:
    """Local graph-style memory with optional sync to a `MemoryEngine`.

    - Stores nodes and edges in-memory for fast local operations
    - Can push nodes to a backing `MemoryEngine` (e.g., UpstashVectorMemoryEngine)
    - Edges are kept locally (vector stores generally lack graph ops)
    """

    def __init__(self, *, engine: Optional[MemoryEngine] = None) -> None:
        self._engine = engine
        self._nodes_by_id: Dict[str, KnowledgeNode] = {}
        self._edges: List[KnowledgeEdge] = []

    # ---- Node operations ----
    def add_node(
        self, node: KnowledgeNode, *, sync: bool = True, context: Optional[str] = None
    ) -> None:
        self._nodes_by_id[node.id] = node
        if sync and self._engine is not None:
            batch_add_nodes(self._engine, [node], context=context)

    def add_nodes(
        self,
        nodes: List[KnowledgeNode],
        *,
        sync: bool = True,
        context: Optional[str] = None,
    ) -> int:
        for n in nodes:
            self._nodes_by_id[n.id] = n
        if sync and self._engine is not None:
            return batch_add_nodes(self._engine, nodes, context=context)
        return len(nodes)

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        return self._nodes_by_id.get(node_id)

    def find_nodes(
        self,
        *,
        category: Optional[KnowledgeCategory | str] = None,
        key: Optional[str] = None,
    ) -> List[KnowledgeNode]:
        out: List[KnowledgeNode] = []
        for n in self._nodes_by_id.values():
            if category and (n.category.value != str(category)):
                continue
            if key and n.key != key:
                continue
            out.append(n)
        return out

    # ---- Edge operations ----
    def add_edge(
        self, edge: KnowledgeEdge, *, sync: bool = True, context: Optional[str] = None
    ) -> None:
        self._edges.append(edge)
        if sync and self._engine is not None:
            # Provide node summaries to improve edge embedding
            batch_add_edges(
                self._engine, [edge], nodes_by_id=self._nodes_by_id, context=context
            )

    def add_edges(
        self,
        edges: List[KnowledgeEdge],
        *,
        sync: bool = True,
        context: Optional[str] = None,
    ) -> int:
        self._edges.extend(edges)
        if sync and self._engine is not None:
            return batch_add_edges(
                self._engine, edges, nodes_by_id=self._nodes_by_id, context=context
            )
        return len(edges)

    def neighbors(self, node_id: str) -> List[KnowledgeNode]:
        neigh_ids = {
            e.target_node_id for e in self._edges if e.source_node_id == node_id
        }
        return [self._nodes_by_id[i] for i in neigh_ids if i in self._nodes_by_id]

    # ---- Sync helpers ----
    def sync_query_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve engine memories relevant to a query for context injection."""
        if not self._engine:
            return []
        try:
            return self._engine.get(query, top_k)
        except Exception:
            return []

    # ---- Engine-scoped listing helpers ----
    def list_engine_memories_for_scope(
        self,
        *,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[str]:
        """Best-effort: list existing engine memories for a given user/unit.

        - Uses vendor-specific fast path when the underlying engine exposes an
          index with metadata filtering (e.g., Upstash Vector).
        - Falls back to a semantic query using a composite text.
        """
        if not self._engine:
            return []
        # Fast path: UpstashVectorMemoryEngine exposes `_index` and optional `_namespace`
        try:
            index = getattr(self._engine, "_index", None)
            if index is not None:
                ns = getattr(self._engine, "_namespace", None)
                filters: List[str] = []
                if ns is not None:
                    filters.append(f"namespace = '{ns}'")
                if user_id:
                    filters.append(f"user_id = '{str(user_id)}'")
                if unit_id:
                    filters.append(f"unit_id = '{str(unit_id)}'")
                filter_expr = " AND ".join(filters) if filters else None
                res = index.query(
                    data=str(user_id or unit_id or "memories"),
                    top_k=limit,
                    include_metadata=True,
                    filter=filter_expr,
                )
                out: List[str] = []
                for item in res or []:
                    md = getattr(item, "metadata", None) or {}
                    cat = md.get("category", "other")
                    key = md.get("key", "note")
                    val = md.get("value", "")
                    if val:
                        out.append(f"{cat}:{key}={val}")
                return out
        except Exception:
            pass
        # Fallback: semantic query with composite context
        try:
            composite = (
                " ".join([p for p in [user_id or "", unit_id or ""] if p]).strip()
                or "memories"
            )
            return self._engine.get(composite, limit)
        except Exception:
            return []

    # ---- Convenience builders ----
    def create_and_add_node(
        self,
        *,
        category: KnowledgeCategory | str = KnowledgeCategory.OTHER,
        key: str = "note",
        value: str = "",
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        confidence: float = 1.0,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        sync: bool = True,
    ) -> KnowledgeNode:
        node = build_node(
            category=category,
            key=key,
            value=value,
            user_id=user_id,
            unit_id=unit_id,
            confidence=confidence,
            context=context,
            metadata=metadata,
            source_id=source_id,
        )
        self.add_node(node, sync=sync, context=context)
        return node

    def create_and_add_edge(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
        relationship: str = "related_to",
        strength: float = 1.0,
        sync: bool = True,
        context: Optional[str] = None,
    ) -> KnowledgeEdge:
        edge = build_edge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship=relationship,
            strength=strength,
        )
        self.add_edge(edge, sync=sync, context=context)
        return edge
