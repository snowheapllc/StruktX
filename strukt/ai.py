from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional

from .config import MiddlewareConfig, StruktConfig, ensure_config_types
from .defaults import (
    GeneralHandler,
    MemoryAugmentedLLMClient,
    SimpleClassifier,
    SimpleLLMClient,
)
from .engine import Engine
from .interfaces import Classifier, Handler, LLMClient, MemoryEngine
from .langchain_helpers import adapt_to_llm_client
from .memory import KnowledgeStore
from .middleware import Middleware
from .types import InvocationState, StruktQueryEnum, StruktResponse, BackgroundTaskInfo
from .utils import coerce_factory, load_factory


class Strukt:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def invoke(self, text: str, context: Dict | None = None) -> StruktResponse:
        state = InvocationState(text=text, context=context or {})
        results = self._engine.run(state)
        responses = [r.response for r in results if r and r.response]
        query_types = [r.status for r in results if r and r.status]
        combined = (
            ". ".join([s.strip().rstrip(". ") for s in responses]) if responses else ""
        )
        if len(query_types) > 1:
            query_type = StruktQueryEnum.MULTIPLE
        elif query_types:
            query_type = query_types[0]
        else:
            query_type = StruktQueryEnum.GENERAL
        return StruktResponse(
            response=combined or "",
            query_type=query_type,
            query_types=query_types,
            parts=list(state.parts or []),
            metadata={},
        )

    async def ainvoke(self, text: str, context: Dict | None = None) -> StruktResponse:
        # Run the sync invoke in a thread to provide true async behavior
        return await asyncio.to_thread(self.invoke, text, context)

    # --- Memory convenience helpers ---
    def get_memory(self) -> MemoryEngine | None:
        return getattr(self._engine, "_memory", None)

    def get_memory_store(self) -> KnowledgeStore | None:
        mem = self.get_memory()
        return getattr(mem, "store", None) if mem is not None else None

    def add_memory(self, text: str, metadata: Dict | None = None) -> None:
        memory = getattr(self._engine, "_memory", None)
        if memory and hasattr(memory, "add"):
            try:
                memory.add(text, metadata or {})  # type: ignore[attr-defined]
            except Exception:
                pass

    def retrieve_memory(self, query: str, top_k: int = 5) -> list[str]:
        memory = getattr(self._engine, "_memory", None)
        if memory and hasattr(memory, "get"):
            try:
                return memory.get(query, top_k)  # type: ignore[attr-defined]
            except Exception:
                return []
        return []

    # --- Background task convenience helpers ---
    def get_background_task_info(self, task_id: str) -> Optional[BackgroundTaskInfo]:
        """Get information about a specific background task."""
        return self._engine.get_background_task_info(task_id)

    def get_all_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all background tasks."""
        return self._engine.get_all_background_tasks()

    def get_background_tasks_by_status(self, status: str) -> List[BackgroundTaskInfo]:
        """Get background tasks filtered by status."""
        return self._engine.get_background_tasks_by_status(status)

    def get_running_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all currently running background tasks."""
        return self._engine.get_running_background_tasks()

    def get_completed_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all completed background tasks."""
        return self._engine.get_completed_background_tasks()

    def get_failed_background_tasks(self) -> List[BackgroundTaskInfo]:
        """Get all failed background tasks."""
        return self._engine.get_failed_background_tasks()


def _build_llm(cfg: StruktConfig) -> LLMClient:
    factory = coerce_factory(cfg.llm.factory)
    if factory is None:
        # Try a sensible default: if OpenAI key is present, attempt ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from langchain_openai import ChatOpenAI

                candidate = ChatOpenAI(api_key=api_key)
                adapted = adapt_to_llm_client(candidate)
                return adapted
            except Exception:
                pass
        return SimpleLLMClient()  # minimal default
    candidate = factory(**cfg.llm.params)  # type: ignore[call-arg]
    # Auto-adapt common LangChain runnables to our LLMClient protocol
    try:
        adapted = adapt_to_llm_client(candidate)
        return adapted
    except Exception:
        return candidate  # type: ignore[return-value]


def _build_classifier(cfg: StruktConfig, llm: LLMClient) -> Classifier:
    factory = coerce_factory(cfg.classifier.factory)
    if factory is None:
        return SimpleClassifier()
    classifier = factory(llm=llm, **cfg.classifier.params)  # type: ignore[call-arg]
    return classifier


def _build_memory(cfg: StruktConfig, llm: LLMClient) -> MemoryEngine | None:
    factory = coerce_factory(cfg.memory.factory)
    if factory is None:
        # Provide a minimal in-memory engine by default for augmentation
        return None
    params = dict(cfg.memory.params or {})
    # Prefer calling without llm; if the factory accepts llm, a 2nd attempt is made
    try:
        return factory(**params)  # type: ignore[misc]
    except TypeError:
        try:
            return factory(llm=llm, **params)  # type: ignore[misc]
        except TypeError:
            return factory(**params)  # type: ignore[misc]


def _build_handlers(
    cfg: StruktConfig, llm: LLMClient, memory: MemoryEngine | None
) -> Dict[str, Handler]:
    handlers: Dict[str, Handler] = {}
    for qtype, factory_like in (cfg.handlers.registry or {}).items():
        factory = coerce_factory(factory_like)
        if factory is None:
            continue
        params = (cfg.handlers.handler_params or {}).get(qtype, {})
        store = getattr(memory, "store", None)
        # Try progressively with richer context while respecting user-provided params
        handler = None
        for pat in (
            {"llm": llm, "memory": memory, "store": store},
            {"llm": llm, "store": store},
            {"llm": llm, "memory": memory},
            {"llm": llm},
        ):
            call_kwargs = dict(params)
            for k, v in pat.items():
                if k not in call_kwargs and v is not None:
                    call_kwargs[k] = v
            try:
                handler = factory(**call_kwargs)
                break
            except TypeError:
                continue
        if handler is None:
            handler = factory(llm=llm, **params)
        handlers[qtype] = handler
    # Ensure a general handler exists
    if StruktQueryEnum.GENERAL not in handlers:
        handlers[StruktQueryEnum.GENERAL] = GeneralHandler(llm=llm)
    return handlers


def _build_middleware(
    config: StruktConfig, *, llm: LLMClient, memory: MemoryEngine | None
) -> list[Middleware]:
    middlewares: list[Middleware] = []
    for item in config.middleware or []:
        # item is normalized to MiddlewareConfig
        mc: MiddlewareConfig = item  # type: ignore[assignment]
        fac = load_factory(mc.factory)
        if fac is None:
            continue
        params = dict(mc.params or {})
        store = getattr(memory, "store", None)
        # Progressive arg patterns; do not override user-supplied params
        patterns = [
            {"llm": llm, "memory": memory, "store": store},
            {"llm": llm, "store": store},
            {"llm": llm, "memory": memory},
            {"store": store},
            {"memory": memory},
            {},
        ]
        mw = None
        for pat in patterns:
            call_kwargs = dict(params)
            for k, v in pat.items():
                if k not in call_kwargs and v is not None:
                    call_kwargs[k] = v
            try:
                mw = fac(**call_kwargs)
                break
            except TypeError:
                continue
        if mw is None:
            # Last resort: no-arg
            mw = fac()
        # Accept duck-typed classes that implement hooks
        middlewares.append(mw)  # type: ignore[arg-type]
    return middlewares


def create(config: StruktConfig) -> Strukt:
    # Normalize config to ensure dicts are coerced into dataclasses
    config = ensure_config_types(config)
    llm = _build_llm(config)
    memory = _build_memory(config, llm)
    # Optionally build a KnowledgeStore bound to the engine
    store = None
    try:
        if getattr(config.memory, "use_store", False) and memory is not None:
            store = KnowledgeStore(engine=memory)
            # expose store on engine for advanced users (duck-typed)
            memory.store = store
    except Exception:
        store = None
    # Wrap LLM with memory augmentation when enabled
    if memory is not None and getattr(config.memory, "augment_llm", True):
        llm = MemoryAugmentedLLMClient(llm, memory)
    classifier = _build_classifier(config, llm)
    handlers = _build_handlers(config, llm, memory)
    middleware = _build_middleware(config, llm=llm, memory=memory)
    engine = Engine(
        classifier=classifier,
        handlers=handlers,
        default_route=config.handlers.default_route,
        memory=memory,
        middleware=middleware,
    )
    return Strukt(engine)
