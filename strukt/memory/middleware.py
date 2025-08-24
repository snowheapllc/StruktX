from __future__ import annotations

import os

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from ..interfaces import LLMClient
from ..memory import KnowledgeStore, build_node
from ..middleware import Middleware
from ..types import HandlerResult, InvocationState


class MemoryExtractionMiddleware(Middleware):
    """Extracts one or more useful memories from the final response and stores them.

    Uses the LLM to transform the input+output into structured memory candidates,
    then adds them to the KnowledgeStore. Prints debug output when STRUKTX_DEBUG=1.
    """

    class MemoryItem(BaseModel):
        category: str = "context"
        key: str = "note"
        value: str = ""
        context: str | None = None

    class MemoryBatch(BaseModel):
        items: list[MemoryExtractionMiddleware.MemoryItem] = []  # type: ignore[name-defined]

    def __init__(self, llm: LLMClient, store: KnowledgeStore | None) -> None:
        self._llm = llm
        self._store = store
        self.console = Console()
        self._debug_enabled = os.getenv("STRUKTX_DEBUG") == "1"

    def after_handle(
        self, state: InvocationState, query_type: str, result: HandlerResult
    ) -> HandlerResult:
        if not self._store:
            return result
        # Run once per invocation
        try:
            flags = state.context.setdefault("_struktx_mw_flags", {})
            if flags.get("memory_extracted_once"):
                return result
            # Set early to avoid multiple runs on the same invocation
            flags["memory_extracted_once"] = True
        except Exception:
            flags = {}
        try:
            # Query existing memories for this user/unit to provide context and prevent duplicates
            user_id = str(state.context.get("user_id", "anon"))
            unit_id = str(state.context.get("unit_id", "default"))
            existing_memories = []
            try:
                all_nodes = self._store.find_nodes()  # type: ignore[attr-defined]
                for n in all_nodes or []:
                    if (
                        str(getattr(n, "user_id", None) or "") == user_id
                        and str(getattr(n, "unit_id", None) or "") == unit_id
                    ):
                        cat = (
                            getattr(n, "category", "").value
                            if hasattr(getattr(n, "category", ""), "value")
                            else str(getattr(n, "category", ""))
                        )
                        key = getattr(n, "key", "")
                        val = getattr(n, "value", "")
                        existing_memories.append(f"{cat}:{key}={val}")
            except Exception:
                existing_memories = []

            # Enrich with engine-backed memories for this scope (best-effort)
            try:
                if hasattr(self._store, "list_engine_memories_for_scope"):
                    engine_mems = self._store.list_engine_memories_for_scope(
                        user_id=user_id, unit_id=unit_id, limit=50
                    )  # type: ignore[attr-defined]
                    for em in engine_mems or []:
                        if em not in existing_memories:
                            existing_memories.append(em)
            except Exception:
                pass

            existing_context = ""
            if existing_memories:
                existing_context = (
                    f"\n\nEXISTING MEMORIES for user {user_id} in unit {unit_id}:\n"
                    + "\n".join(f"- {m}" for m in existing_memories)
                    + "\n\nDO NOT extract memories that are duplicates or very similar to the existing ones above."
                )

            prompt = (
                "You will extract useful, durable memory entries from a user interaction.\n"
                "Existing memories (avoid duplicates):\n{existing_context}\n"
                "Input text: {text}\n"
                "Final response: {response}\n\n"
                "STRICT CRITERIA:\n"
                "- Only extract durable user information (e.g., preferences, recurring behaviors, stable locations).\n"
                "- Do NOT extract transient questions, requests, or external facts (e.g., 'best place...', 'what is...').\n"
                "- Do NOT extract memories that duplicate or closely resemble existing memories listed above.\n"
                "- If nothing durable is present, return an empty list.\n\n"
                "Return JSON with an array 'items', where each item has: category (one of location, preference, behavior, context, other),\n"
                "key (short identifier), value (brief content), and optional context. If none, return items: [].\n"
            )
            payload = {
                "text": state.text,
                "response": result.response,
                "existing_context": existing_context,
            }
            out = self._llm.structured(
                prompt.format(**payload),
                MemoryExtractionMiddleware.MemoryBatch,
                context=state.context,
                query_hint=state.text,
                augment_source="middleware.memory_extraction",
            )
            items = list(getattr(out, "items", []) or [])

            debug_rows = []
            seen_pairs: set[tuple[str, str, str, str]] = (
                set()
            )  # (cat, value_lower, user_id, unit_id)
            all_cats = {"location", "preference", "behavior", "context", "other"}
            # Only allow extracting values that appear in input, output, or existing context
            guard_text = f"{state.text}\n{result.response}\n{existing_context}".lower()
            for item in items:
                try:
                    cat = (item.category or "context").strip().lower()
                    if cat not in {
                        "location",
                        "preference",
                        "behavior",
                        "context",
                        "other",
                    }:
                        cat = "context"
                    key = (item.key or "note").strip()
                    val = (item.value or "").strip()
                    if not val:
                        continue
                    if val.lower() not in guard_text:
                        # Skip hallucinated values that are not grounded in input/response/context
                        continue
                    ctx = (item.context or "").strip()
                    # Heuristics to discard non-durable/question-like items
                    low_key = key.lower()
                    low_val = val.lower()
                    if val.endswith("?"):
                        continue
                    if low_val.startswith(
                        ("what ", "where ", "when ", "how ", "who ", "which ")
                    ):
                        continue
                    if "best " in low_key or "best " in low_val:
                        continue
                    if cat == "location" and ("best " in low_val or "place" in low_val):
                        continue
                    # Duplicate guard:
                    sig = (cat, low_val, user_id, unit_id)
                    if sig in seen_pairs:
                        continue
                    seen_pairs.add(sig)
                    # Check existing nodes across relevant categories, ignoring key differences
                    is_dup = False
                    try:
                        categories_to_check = all_cats if cat in all_cats else {cat}
                        for c in categories_to_check:
                            existing = self._store.find_nodes(category=c)  # type: ignore[attr-defined]
                            for n in existing or []:
                                if (
                                    (getattr(n, "value", "").strip().lower() == low_val)
                                    and str(getattr(n, "user_id", None) or "")
                                    == user_id
                                    and str(getattr(n, "unit_id", None) or "")
                                    == unit_id
                                ):
                                    is_dup = True
                                    break
                            if is_dup:
                                break
                    except Exception:
                        is_dup = False
                    # Strict duplicate skip: if value appears in existing_context, skip
                    if (f"={low_val}" in existing_context.lower()) or is_dup:
                        if self._debug_enabled:
                            self.console.print(
                                f"[dim]Skipping duplicate memory for {user_id}/{unit_id}: {cat}:{key}={val}[/dim]"
                            )
                        continue
                    node = build_node(
                        category=cat,
                        key=key,
                        value=val,
                        context=ctx,
                        user_id=user_id,
                        unit_id=unit_id,
                    )
                    self._store.add_node(node, sync=True)
                    debug_rows.append((cat, key, val))
                except Exception:
                    continue

            if os.getenv("STRUKTX_DEBUG") and debug_rows:
                table = Table(
                    title="🧠 Extracted Memories",
                    show_header=True,
                    header_style="bold magenta",
                )
                table.add_column("Category", style="cyan")
                table.add_column("Key", style="yellow")
                table.add_column("Value", style="green")
                for cat, key, val in debug_rows:
                    table.add_row(
                        cat, key, (val[:60] + "...") if len(val) > 60 else val
                    )
                self.console.print(table)
            # Already marked as processed above
        except Exception:
            return result
        return result
