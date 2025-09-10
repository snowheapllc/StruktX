from __future__ import annotations

import asyncio
import json
import os
from typing import Dict, List, Optional

from .config import MiddlewareConfig, StruktConfig, ensure_config_types
from .defaults import (
    GeneralHandler,
    MemoryAugmentedLLMClient,
    SimpleClassifier,
    SimpleLLMClient,
    UniversalLLMLogger,
)
from .engine import Engine
from .interfaces import Classifier, Handler, LLMClient, MemoryEngine
from .langchain_helpers import adapt_to_llm_client
from .memory import KnowledgeStore
from .middleware import Middleware
from .types import InvocationState, StruktQueryEnum, StruktResponse, BackgroundTaskInfo
from .utils import coerce_factory, load_factory
from .logging import get_logger, StruktLogger
from .tracing import (
    init_otel,
    enable_global_tracing,
)


class Strukt:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def invoke(self, text: str, context: Dict | None = None) -> StruktResponse:
        state = InvocationState(text=text, context=context or {})
        results = self._engine.run(state)
        responses = [r.response for r in results if r and r.response]
        query_types = [r.status for r in results if r and r.status]
        # Handle both string and dict responses
        if len(responses) == 1 and isinstance(responses[0], dict):
            # For single dict response, return the entire dict structure
            combined = responses[0]
        else:
            # For multiple responses or string responses, process as before
            processed_responses = []
            for s in responses:
                if isinstance(s, dict):
                    # For dict responses in multi-response scenarios, extract message or convert to JSON
                    if "message" in s and isinstance(s["message"], str):
                        processed_responses.append(s["message"].strip().rstrip(". "))
                    else:
                        processed_responses.append(json.dumps(s))
                elif isinstance(s, str):
                    processed_responses.append(s.strip().rstrip(". "))
                else:
                    processed_responses.append(str(s))

            combined = ". ".join(processed_responses) if processed_responses else ""
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

    # --- Weave logging convenience helpers ---
    def get_logger(self) -> StruktLogger:
        """Get the StruktX logger instance."""
        return get_logger("struktx")

    def is_weave_available(self) -> bool:
        """Check if Weave logging is available and initialized."""
        logger = self.get_logger()
        return logger.is_weave_available()

    def get_weave_project_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get the current Weave project name and environment."""
        logger = self.get_logger()
        return logger.get_weave_project_info()

    def create_weave_op(self, func=None, name=None, call_display_name=None):
        """Create a Weave operation decorator for tracking function calls."""
        logger = self.get_logger()
        return logger.create_weave_op(func, name, call_display_name)

    def weave_context(
        self,
        user_id: Optional[str] = None,
        unit_id: Optional[str] = None,
        unit_name: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """Context manager for Weave logging with user context.

        Args:
            user_id: Explicit user ID (takes precedence over context)
            unit_id: Explicit unit ID (takes precedence over context)
            unit_name: Explicit unit name (takes precedence over context)
            context: Dictionary to extract user context from (fallback if explicit values not provided)
        """
        logger = self.get_logger()
        return logger.weave_context(user_id, unit_id, unit_name, context)

    def weave_context_from_state(self, state):
        """Context manager that automatically extracts user context from InvocationState.

        Args:
            state: InvocationState object containing context information
        """
        logger = self.get_logger()
        return logger.weave_context_from_state(state)


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
                return UniversalLLMLogger(adapted)
            except Exception:
                pass
        return UniversalLLMLogger(SimpleLLMClient())  # minimal default with logging
    candidate = factory(**cfg.llm.params)  # type: ignore[call-arg]
    # Auto-adapt common LangChain runnables to our LLMClient protocol
    try:
        adapted = adapt_to_llm_client(candidate)
        # Wrap with universal LLM logger for comprehensive logging
        return UniversalLLMLogger(adapted)
    except Exception:
        # Wrap even the fallback candidate
        return UniversalLLMLogger(candidate)  # type: ignore[arg-type]


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

    # Initialize Weave logging if enabled
    if config.weave.enabled:
        logger = get_logger("struktx")
        logger.init_weave(
            project_name=config.weave.project_name, environment=config.weave.environment
        )

        # Enable global unified tracing
        enable_global_tracing()

        # Auto-instrument all StruktX base classes and their methods (disabled for now)
        # auto_instrument_struktx()

    # Initialize OpenTelemetry (export to Weave OTLP) if enabled
    if getattr(config, "opentelemetry", None):
        try:
            init_otel(config.opentelemetry)
        except Exception:
            pass

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
        weave_config=config.weave,
        tracing_config=config.tracing,
        evaluation_config=config.evaluation,
    )
    return Strukt(engine)
