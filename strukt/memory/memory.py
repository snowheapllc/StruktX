from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces import MemoryEngine
from .memory_types import KnowledgeCategory, KnowledgeEdge, KnowledgeNode


class UpstashVectorMemoryEngine(MemoryEngine):
    """Generic Upstash Vector knowledge-graph-like memory engine.

    Stores nodes as vectors with:
      data: a structured text like "Category: X | Key: k | Value: v | Context: ..."
      metadata: generic free-form fields including ids and namespace

    Edges are not persisted in Upstash (no native graph), but can be embedded
    into node metadata if needed by the application.

    Configuration:
    - index_url/index_token (env fallback): Upstash credentials
    - namespace: optional string to partition data
    - default_user_id/unit_id: optional default scoping fields
    - metadata_filter: {k: v} applied as AND filter in queries
    """

    def __init__(
        self,
        *,
        index_url: str | None = None,
        index_token: str | None = None,
        namespace: str | None = None,
        default_user_id: str | None = None,
        default_unit_id: str | None = None,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> None:
        self._configured = False
        self._namespace = namespace
        self._metadata_filter = dict(metadata_filter or {})
        self._default_user_id = default_user_id
        self._default_unit_id = default_unit_id
        try:
            import os

            from upstash_vector import Index  # type: ignore

            url = index_url or os.getenv("UPSTASH_VECTOR_REST_URL")
            token = index_token or os.getenv("UPSTASH_VECTOR_REST_TOKEN")
            if not url or not token:
                raise ValueError("Upstash Vector credentials missing")
            self._index = Index(url=url, token=token)
            self._configured = True
        except Exception:
            self._configured = False
            self._index = None  # type: ignore[assignment]

    def _build_metadata(
        self, node: KnowledgeNode, context: str | None, metadata: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        m = {
            "node_id": node.id,
            "category": node.category.value,
            "key": node.key,
            "value": node.value,
            "user_id": node.user_id or self._default_user_id,
            "unit_id": node.unit_id or self._default_unit_id,
            "confidence": node.confidence,
            "timestamp": node.timestamp.isoformat(),
        }
        if context:
            m["context"] = context
        if metadata:
            m.update(metadata)
        if self._namespace is not None:
            m["namespace"] = self._namespace
        return m

    def _build_filter(self) -> str | None:
        parts: List[str] = []
        if self._namespace is not None:
            parts.append(f"namespace = '{self._namespace}'")
        for k, v in (self._metadata_filter or {}).items():
            # Only simple equality filters are supported here
            vs = str(v).replace("'", "'")
            parts.append(f"{k} = '{vs}'")
        return " AND ".join(parts) if parts else None

    # --- helpers to reduce method complexity ---
    def _create_node(self, text: str, metadata: Dict[str, Any] | None) -> KnowledgeNode:
        md = metadata or {}
        return KnowledgeNode(
            category=KnowledgeCategory(md.get("category", "other")),
            key=str(md.get("key", "note")),
            value=str(md.get("value", text)),
            user_id=md.get("user_id", self._default_user_id),
            unit_id=md.get("unit_id", self._default_unit_id),
            confidence=float(md.get("confidence", 1.0)),
            metadata=(
                {md.get("meta_key", ""): md.get("meta_val", "")} if metadata else {}
            ),
            source_id=md.get("source_id"),
        )

    def _vector_from_node(
        self,
        node: KnowledgeNode,
        context: Optional[str],
        metadata: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        return {
            "id": (metadata or {}).get("id")
            or f"{node.user_id or 'anon'}:{node.unit_id or 'default'}:{node.id}",
            "data": f"Category: {node.category.value} | Key: {node.key} | Value: {node.value} | Context: {context or ''}",
            "metadata": self._build_metadata(node, context, metadata),
        }

    def _format_hit(self, item: Any) -> str:
        md = getattr(item, "metadata", None) or {}
        cat = md.get("category", "other")
        key = md.get("key", "note")
        val = md.get("value", "")
        return f"{cat}:{key}={val}"

    def _query_ids_by_text(self, text: str, top_k: int = 20) -> List[str]:
        filter_expr = self._build_filter()
        res = self._index.query(
            data=text, top_k=top_k, include_metadata=True, filter=filter_expr
        )
        ids = [getattr(item, "id", None) for item in (res or [])]
        return [i for i in ids if i]

    # MemoryEngine API
    def add(self, text: str, metadata: Dict[str, Any] | None = None) -> None:
        if not self._configured or not self._index:
            return
        try:
            node = self._create_node(text, metadata)
            context = (metadata or {}).get("context")
            vector = self._vector_from_node(node, context, metadata)
            # Upsert single vector
            self._index.upsert(vectors=[vector])
        except Exception:
            return

    def get(self, query: str, top_k: int = 5) -> List[str]:
        if not self._configured or not self._index:
            return []
        try:
            filter_expr = self._build_filter()
            res = self._index.query(
                data=query, top_k=top_k, include_metadata=True, filter=filter_expr
            )
            return [self._format_hit(item) for item in (res or [])]
        except Exception:
            return []

    def get_scoped(
        self,
        query: str,
        *,
        user_id: str | None = None,
        unit_id: str | None = None,
        top_k: int = 5,
    ) -> List[str]:
        if not self._configured or not self._index:
            return []
        try:
            # Build combined filter: base filter + optional scope
            filters: List[str] = []
            base = self._build_filter()
            if base:
                filters.append(base)
            if user_id:
                filters.append(f"user_id = '{str(user_id)}'")
            if unit_id:
                filters.append(f"unit_id = '{str(unit_id)}'")
            filter_expr = " AND ".join(filters) if filters else None
            res = self._index.query(
                data=query, top_k=top_k, include_metadata=True, filter=filter_expr
            )
            return [self._format_hit(item) for item in (res or [])]
        except Exception:
            # Fallback to unscoped
            return self.get(query, top_k)

    def remove(self, text: str) -> int:
        if not self._configured or not self._index:
            return 0
        try:
            # Find likely matches then delete by ids
            ids = self._query_ids_by_text(text, top_k=20)
            if not ids:
                return 0
            self._index.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def cleanup(self, **kwargs: Any) -> Dict[str, Any]:
        # Best-effort: no server-side TTL; return an informative payload
        return {
            "ok": self._configured,
            "reason": "Upstash Vector does not support generic TTL deletes via REST; implement app-specific policy using metadata filters.",
        }


# --------------------- Helper builders and batch operations ---------------------


def build_node(
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
) -> KnowledgeNode:
    cat = (
        category
        if isinstance(category, KnowledgeCategory)
        else KnowledgeCategory(str(category))
    )
    node = KnowledgeNode(
        category=cat,
        key=key,
        value=value,
        user_id=user_id,
        unit_id=unit_id,
        confidence=confidence,
        metadata=dict(metadata or {}),
        source_id=source_id,
    )
    if context:
        node.metadata.setdefault("context", context)
    return node


def build_edge(
    *,
    source_node_id: str,
    target_node_id: str,
    relationship: str = "related_to",
    strength: float = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relationship=relationship,
        strength=strength,
    )


def batch_add_nodes(
    engine: MemoryEngine,
    nodes: List[KnowledgeNode],
    *,
    context: Optional[str] = None,
) -> int:
    """Add a batch of nodes to any MemoryEngine by serializing nodes to text.

    Engines that support metadata can use it to store rich fields; simple engines
    will at least persist the formatted text.
    """
    added = 0
    for node in nodes:
        text = f"Category: {node.category.value} | Key: {node.key} | Value: {node.value} | Context: {context or node.metadata.get('context', '')}"
        md: Dict[str, Any] = {
            "id": f"{node.user_id or 'anon'}:{node.unit_id or 'default'}:{node.id}",
            "node_id": node.id,
            "category": node.category.value,
            "key": node.key,
            "value": node.value,
            "user_id": node.user_id,
            "unit_id": node.unit_id,
            "confidence": node.confidence,
            "timestamp": node.timestamp.isoformat(),
        }
        # Merge additional metadata
        for k, v in (node.metadata or {}).items():
            if k not in md:
                md[k] = v
        if context and "context" not in md:
            md["context"] = context
        try:
            engine.add(text, md)
            added += 1
        except Exception:
            continue
    return added


def batch_add_edges(
    engine: MemoryEngine,
    edges: List[KnowledgeEdge],
    *,
    nodes_by_id: Optional[Dict[str, KnowledgeNode]] = None,
    context: Optional[str] = None,
) -> int:
    """Add a batch of edges by serializing them to text with metadata links.

    - Each edge is stored as a separate vector with type='edge'
    - Uses source/target summaries (if nodes provided) to improve embedding quality
    """

    def _summaries(e: KnowledgeEdge) -> tuple[str, str]:
        src = nodes_by_id.get(e.source_node_id) if nodes_by_id else None
        tgt = nodes_by_id.get(e.target_node_id) if nodes_by_id else None
        src_summary = (
            f"{src.category.value}:{src.key}={src.value}" if src else e.source_node_id
        )
        tgt_summary = (
            f"{tgt.category.value}:{tgt.key}={tgt.value}" if tgt else e.target_node_id
        )
        return src_summary, tgt_summary

    def _serialize(e: KnowledgeEdge) -> tuple[str, Dict[str, Any]]:
        src_summary, tgt_summary = _summaries(e)
        text = (
            f"Relationship: {e.relationship} | Strength: {e.strength} | "
            f"Source: {src_summary} | Target: {tgt_summary} | Context: {context or ''}"
        )
        md: Dict[str, Any] = {
            "type": "edge",
            "edge_id": f"{e.source_node_id}:{e.relationship}:{e.target_node_id}",
            "source_node_id": e.source_node_id,
            "target_node_id": e.target_node_id,
            "relationship": e.relationship,
            "strength": e.strength,
            "timestamp": e.timestamp.isoformat(),
        }
        if context:
            md.setdefault("context", context)
        return text, md

    added = 0
    for edge in edges:
        try:
            text, md = _serialize(edge)
            engine.add(text, md)
            added += 1
        except Exception:
            continue
    return added


def get_edges_for_node(
    engine: MemoryEngine,
    *,
    node_id: str,
    relationship: Optional[str] = None,
    top_k: int = 20,
) -> List[str]:
    """Retrieve human-readable edge summaries for a node.

    Prefers metadata-filtered index queries when available; otherwise falls back
    to semantic search using node id (and relationship if provided).
    """

    def _edge_item_to_text(item: Any) -> Optional[str]:
        md = getattr(item, "metadata", None) or {}
        if md.get("type") != "edge":
            return None
        if relationship and md.get("relationship") != relationship:
            return None
        rel = md.get("relationship", "related_to")
        src = md.get("source_node_id", "?")
        tgt = md.get("target_node_id", "?")
        return f"{src} -[{rel}]-> {tgt}"

    def _query_index(index: Any, filter_expr: str) -> List[str]:
        res = index.query(
            data=node_id, top_k=top_k, include_metadata=True, filter=filter_expr
        )
        return [
            txt for txt in (_edge_item_to_text(item) for item in (res or [])) if txt
        ]

    # Fast path: UpstashVectorMemoryEngine exposes `_index` and optional `_namespace`
    index = getattr(engine, "_index", None)
    if index is not None:
        try:
            ns = getattr(engine, "_namespace", None)
            base = (
                f"type = 'edge' AND namespace = '{ns}'"
                if ns is not None
                else "type = 'edge'"
            )
            filters = [
                base + f" AND source_node_id = '{node_id}'",
                base + f" AND target_node_id = '{node_id}'",
            ]
            results: List[str] = []
            for f in filters:
                results.extend(_query_index(index, f))
            return results
        except Exception:
            pass

    # Fallback: semantic search
    query = node_id if not relationship else f"{node_id} {relationship}"
    try:
        return engine.get(query, top_k)
    except Exception:
        return []
