from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

# Factories may be callables or import strings; resolved at create().
Factory = Union[str, Callable[..., Any]]


@dataclass
class LLMClientConfig:
    factory: Optional[Factory] = None  # if None, use a minimal default lc client
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    factory: Optional[Factory] = None  # must produce a Classifier instance
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    factory: Optional[Factory] = None  # memory engine factory
    params: Dict[str, Any] = field(default_factory=dict)  # engine params and toggles
    # Optional toggles (also readable via params for flexibility)
    use_store: bool = False  # if true, build a KnowledgeStore bound to the engine
    augment_llm: bool = True  # if true, wrap LLM with memory augmentation


@dataclass
class HandlersConfig:
    registry: Dict[str, Factory] = field(
        default_factory=dict
    )  # query_type -> handler factory/callable/import
    handler_params: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # query_type -> params for factory
    default_route: Optional[str] = None  # fallback type name


@dataclass
class HumanLayerConfig:
    enabled: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtrasConfig:
    humanlayer: HumanLayerConfig = field(default_factory=HumanLayerConfig)


@dataclass
class MiddlewareConfig:
    factory: Factory | None = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StruktConfig:
    llm: LLMClientConfig = field(default_factory=LLMClientConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    handlers: HandlersConfig = field(default_factory=HandlersConfig)
    extras: ExtrasConfig = field(default_factory=ExtrasConfig)
    middleware: List[MiddlewareConfig] = field(default_factory=list)


# Helper functions to coerce user-friendly dict inputs into dataclasses
def _coerce_llm_config(value: Any) -> LLMClientConfig:
    if isinstance(value, LLMClientConfig):
        return value
    if isinstance(value, dict):
        return LLMClientConfig(**value)
    return LLMClientConfig()


def _coerce_classifier_config(value: Any) -> ClassifierConfig:
    if isinstance(value, ClassifierConfig):
        return value
    if isinstance(value, dict):
        return ClassifierConfig(**value)
    return ClassifierConfig()


def _coerce_memory_config(value: Any) -> MemoryConfig:
    if isinstance(value, MemoryConfig):
        return value
    if isinstance(value, dict):
        return MemoryConfig(**value)
    return MemoryConfig()


def _coerce_handlers_config(value: Any) -> HandlersConfig:
    if isinstance(value, HandlersConfig):
        return value
    if isinstance(value, dict):
        return HandlersConfig(**value)
    return HandlersConfig()


def _coerce_extras_config(value: Any) -> ExtrasConfig:
    if isinstance(value, ExtrasConfig):
        return value
    if isinstance(value, dict):
        # Support nested humanlayer dict as well
        humanlayer = value.get("humanlayer") if isinstance(value, dict) else None
        if isinstance(humanlayer, HumanLayerConfig):
            coerced_hl = humanlayer
        elif isinstance(humanlayer, dict):
            coerced_hl = HumanLayerConfig(**humanlayer)
        else:
            coerced_hl = HumanLayerConfig()
        return ExtrasConfig(humanlayer=coerced_hl)
    return ExtrasConfig()


def ensure_config_types(config: StruktConfig) -> StruktConfig:
    """Normalize a `StruktConfig` instance so all nested fields are dataclasses.

    Accepts cases where callers passed dicts for sub-configs and coerces them.
    Mutates and returns the provided `config` for convenience.
    """
    # Coerce each sub-config that may have been provided as a dict
    config.llm = _coerce_llm_config(getattr(config, "llm", None))
    config.classifier = _coerce_classifier_config(getattr(config, "classifier", None))
    config.memory = _coerce_memory_config(getattr(config, "memory", None))
    config.handlers = _coerce_handlers_config(getattr(config, "handlers", None))
    config.extras = _coerce_extras_config(getattr(config, "extras", None))
    # Normalize middleware list
    config.middleware = _coerce_middleware_list(getattr(config, "middleware", []))
    return config


def _coerce_middleware_list(value: Any) -> List[MiddlewareConfig]:
    items: List[MiddlewareConfig] = []
    if not isinstance(value, list):
        return items
    for item in value:
        if isinstance(item, MiddlewareConfig):
            items.append(item)
        elif isinstance(item, dict):
            items.append(MiddlewareConfig(**item))
        else:
            # callable or import string
            items.append(MiddlewareConfig(factory=item, params={}))
    return items
