from __future__ import annotations

from typing import Dict, List
import asyncio

from .config import StruktConfig
from .interfaces import Classifier, Handler, LLMClient, MemoryEngine
from .types import StruktResponse, InvocationState
from .engine import Engine
from .utils import load_factory
from .middleware import Middleware
from .defaults import SimpleClassifier, GeneralHandler, SimpleLLMClient


class Strukt:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def invoke(self, text: str, context: Dict | None = None) -> StruktResponse:
        state = InvocationState(text=text, context=context or {})
        results = self._engine.run(state)
        responses = [r.response for r in results if r and r.response]
        query_types = [r.status for r in results if r and r.status]
        combined = ". ".join([s.strip().rstrip(". ") for s in responses]) if responses else ""
        if len(query_types) > 1:
            query_type = "MULTIPLE"
        elif query_types:
            query_type = query_types[0]
        else:
            query_type = "GENERAL"
        return StruktResponse(
            response=combined or "",
            query_type=query_type,
            query_types=query_types,
            parts=list(state.parts or []),
            metadata={}
        )

    async def ainvoke(self, text: str, context: Dict | None = None) -> StruktResponse:
        # Run the sync invoke in a thread to provide true async behavior
        return await asyncio.to_thread(self.invoke, text, context)


def _build_llm(cfg: StruktConfig) -> LLMClient:
    factory = load_factory(cfg.llm.factory)
    if factory is None:
        return SimpleLLMClient()  # minimal default
    return factory(**cfg.llm.params)  # type: ignore[call-arg]


def _build_classifier(cfg: StruktConfig, llm: LLMClient) -> Classifier:
    factory = load_factory(cfg.classifier.factory)
    if factory is None:
        return SimpleClassifier()
    return factory(llm=llm, **cfg.classifier.params)  # type: ignore[call-arg]


def _build_memory(cfg: StruktConfig, llm: LLMClient) -> MemoryEngine | None:
    factory = load_factory(cfg.memory.factory)
    if factory is None:
        return None
    return factory(llm=llm, **cfg.memory.params)  # type: ignore[call-arg]


def _build_handlers(cfg: StruktConfig, llm: LLMClient) -> Dict[str, Handler]:
    handlers: Dict[str, Handler] = {}
    for qtype, factory_like in (cfg.handlers.registry or {}).items():
        factory = load_factory(factory_like)
        if factory is None:
            continue
        params = (cfg.handlers.handler_params or {}).get(qtype, {})
        handler = factory(llm=llm, **params)
        handlers[qtype] = handler
    # Ensure a general handler exists
    if "general" not in handlers:
        handlers["general"] = GeneralHandler()
    return handlers


def _build_middleware(config: StruktConfig) -> list[Middleware]:
    middlewares: list[Middleware] = []
    for fac_like in (config.middleware or []):
        fac = load_factory(fac_like)
        if fac is None:
            continue
        mw = fac()
        # Accept duck-typed classes that implement hooks
        middlewares.append(mw)  # type: ignore[arg-type]
    return middlewares


def create(config: StruktConfig) -> Strukt:
    llm = _build_llm(config)
    classifier = _build_classifier(config, llm)
    memory = _build_memory(config, llm)
    handlers = _build_handlers(config, llm)
    middleware = _build_middleware(config)
    engine = Engine(
        classifier=classifier,
        handlers=handlers,
        default_route=config.handlers.default_route,
        memory=memory,
        middleware=middleware,
    )
    return Strukt(engine)


