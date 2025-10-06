from __future__ import annotations

import json
import os

from .config import MiddlewareConfig, StruktConfig, ensure_config_types
from .defaults import (
    GeneralHandler,
    MemoryAugmentedLLMClient,
    SimpleClassifier,
    SimpleLLMClient,
    UniversalLLMLogger,
)
from .retry import RetryConfig
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

    def invoke(self, text: str, context: dict | None = None) -> StruktResponse:
        state = InvocationState(text=text, context=context or {})
        results = self._engine.run(state)
        responses = [r.response for r in results if r and r.response]
        query_types = [r.status for r in results if r and r.status]
        # Handle both string and dict responses
        if len(responses) == 1 and isinstance(responses[0], dict):
            # For single dict response, return the entire dict structure
            combined = responses[0]
        elif len(responses) > 1:
            # For multiple responses, check if all are dicts (Pydantic objects)
            all_dicts = all(isinstance(r, dict) for r in responses)

            if all_dicts:
                # When all responses are structured objects, return them as a list
                combined = list(responses)
            else:
                # When there's a mix or all strings, process as strings
                processed_responses = []
                for s in responses:
                    if isinstance(s, dict):
                        # For dict responses in mixed scenarios, extract message or convert to JSON
                        if "message" in s and isinstance(s["message"], str):
                            processed_responses.append(
                                s["message"].strip().rstrip(". ")
                            )
                        else:
                            processed_responses.append(json.dumps(s))
                    elif isinstance(s, str):
                        processed_responses.append(s.strip().rstrip(". "))
                    else:
                        processed_responses.append(str(s))

                combined = ". ".join(processed_responses) if processed_responses else ""
        else:
            combined = ""
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

    async def ainvoke(self, text: str, context: dict | None = None) -> StruktResponse:
        """Async invoke using the engine's native async implementation."""
        state = InvocationState(text=text, context=context or {})
        results = await self._engine.arun(state)
        responses = [r.response for r in results if r and r.response]
        query_types = [r.status for r in results if r and r.status]
        # Handle both string and dict responses
        if len(responses) == 1 and isinstance(responses[0], dict):
            # For single dict response, return the entire dict structure
            combined = responses[0]
        elif len(responses) > 1:
            # For multiple responses, check if all are dicts (Pydantic objects)
            all_dicts = all(isinstance(r, dict) for r in responses)

            if all_dicts:
                # When all responses are structured objects, return them as a list
                combined = list(responses)
            else:
                # When there's a mix or all strings, process as strings
                processed_responses = []
                for s in responses:
                    if isinstance(s, dict):
                        # For dict responses in mixed scenarios, extract message or convert to JSON
                        if "message" in s and isinstance(s["message"], str):
                            processed_responses.append(
                                s["message"].strip().rstrip(". ")
                            )
                        else:
                            processed_responses.append(json.dumps(s))
                    elif isinstance(s, str):
                        processed_responses.append(s.strip().rstrip(". "))
                    else:
                        processed_responses.append(str(s))

                combined = ". ".join(processed_responses) if processed_responses else ""
        else:
            combined = ""
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

    # --- Memory convenience helpers ---
    def get_memory(self) -> MemoryEngine | None:
        return getattr(self._engine, "_memory", None)

    def get_memory_store(self) -> KnowledgeStore | None:
        mem = self.get_memory()
        return getattr(mem, "store", None) if mem is not None else None

    def add_memory(self, text: str, metadata: dict | None = None) -> None:
        memory = getattr(self._engine, "_memory", None)
        if memory and hasattr(memory, "add"):
            try:
                memory.add(text, metadata or {})  # type: ignore[attr-defined]
            except Exception:
                pass

    async def aadd_memory(self, text: str, metadata: dict | None = None) -> None:
        """Async add memory."""
        memory = getattr(self._engine, "_memory", None)
        if memory and hasattr(memory, "aadd"):
            try:
                await memory.aadd(text, metadata or {})  # type: ignore[attr-defined]
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

    async def aretrieve_memory(self, query: str, top_k: int = 5) -> list[str]:
        """Async retrieve memory."""
        memory = getattr(self._engine, "_memory", None)
        if memory and hasattr(memory, "aget"):
            try:
                return await memory.aget(query, top_k)  # type: ignore[attr-defined]
            except Exception:
                return []
        return []

    # --- Background task convenience helpers ---
    def get_background_task_info(self, task_id: str) -> BackgroundTaskInfo | None:
        """Get information about a specific background task."""
        return self._engine.get_background_task_info(task_id)

    def get_all_background_tasks(self) -> list[BackgroundTaskInfo]:
        """Get all background tasks."""
        return self._engine.get_all_background_tasks()

    def get_background_tasks_by_status(self, status: str) -> list[BackgroundTaskInfo]:
        """Get background tasks filtered by status."""
        return self._engine.get_background_tasks_by_status(status)

    def get_running_background_tasks(self) -> list[BackgroundTaskInfo]:
        """Get all currently running background tasks."""
        return self._engine.get_running_background_tasks()

    def get_completed_background_tasks(self) -> list[BackgroundTaskInfo]:
        """Get all completed background tasks."""
        return self._engine.get_completed_background_tasks()

    def get_failed_background_tasks(self) -> list[BackgroundTaskInfo]:
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

    def get_weave_project_info(self) -> tuple[str | None, str | None]:
        """Get the current Weave project name and environment."""
        logger = self.get_logger()
        return logger.get_weave_project_info()

    def create_weave_op(self, func=None, name=None, call_display_name=None):
        """Create a Weave operation decorator for tracking function calls."""
        logger = self.get_logger()
        return logger.create_weave_op(func, name, call_display_name)

    def weave_context(
        self,
        user_id: str | None = None,
        unit_id: str | None = None,
        unit_name: str | None = None,
        context: dict | None = None,
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
    # Create retry config if specified
    retry_config = None
    if cfg.llm.retry:
        retry_config = RetryConfig(**cfg.llm.retry)

    factory = coerce_factory(cfg.llm.factory)
    if factory is None:
        # Try a sensible default: if OpenAI key is present, attempt ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from langchain_openai import ChatOpenAI

                candidate = ChatOpenAI(api_key=api_key)
                adapted = adapt_to_llm_client(candidate)
                base_client = UniversalLLMLogger(adapted, retry_config)
            except Exception:
                base_client = UniversalLLMLogger(SimpleLLMClient(), retry_config)
        else:
            base_client = UniversalLLMLogger(SimpleLLMClient(), retry_config)
    else:
        candidate = factory(**cfg.llm.params)  # type: ignore[call-arg]
        # Auto-adapt common LangChain runnables to our LLMClient protocol
        try:
            adapted = adapt_to_llm_client(candidate)
            base_client = UniversalLLMLogger(adapted, retry_config)
        except Exception:
            base_client = UniversalLLMLogger(candidate, retry_config)  # type: ignore[arg-type]

    # Wrap with OptimizedLLMClient if optimizations are enabled
    if cfg.optimizations and (
        cfg.optimizations.enable_llm_streaming
        or cfg.optimizations.enable_llm_batching
        or cfg.optimizations.enable_llm_caching
    ):
        from .llm_clients import OptimizedLLMClient

        return OptimizedLLMClient(
            base_client,
            enable_streaming=cfg.optimizations.enable_llm_streaming,
            enable_batching=cfg.optimizations.enable_llm_batching,
            enable_caching=cfg.optimizations.enable_llm_caching,
            batch_size=cfg.optimizations.llm_batch_size,
            cache_size=cfg.optimizations.llm_cache_size,
            ttl=cfg.optimizations.llm_cache_ttl,
        )

    return base_client


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
) -> dict[str, Handler]:
    handlers: dict[str, Handler] = {}
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

    # Get optimization config
    opt_config = config.optimizations if hasattr(config, "optimizations") else None

    engine = Engine(
        classifier=classifier,
        handlers=handlers,
        default_route=config.handlers.default_route,
        memory=memory,
        middleware=middleware,
        weave_config=config.weave,
        tracing_config=config.tracing,
        evaluation_config=config.evaluation,
        max_concurrent_handlers=opt_config.max_concurrent_handlers
        if opt_config
        else 10,
        enable_performance_monitoring=opt_config.enable_performance_monitoring
        if opt_config
        else True,
    )
    return Strukt(engine)
