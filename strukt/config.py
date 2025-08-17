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
    factory: Optional[Factory] = None  # if None, no memory
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlersConfig:
    registry: Dict[str, Factory] = field(default_factory=dict)  # query_type -> handler factory/callable/import
    handler_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # query_type -> params for factory
    default_route: Optional[str] = None  # fallback type name


@dataclass
class HumanLayerConfig:
    enabled: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtrasConfig:
    humanlayer: HumanLayerConfig = field(default_factory=HumanLayerConfig)


@dataclass
class StruktConfig:
    llm: LLMClientConfig = field(default_factory=LLMClientConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    handlers: HandlersConfig = field(default_factory=HandlersConfig)
    extras: ExtrasConfig = field(default_factory=ExtrasConfig)
    middleware: List[Factory] = field(default_factory=list)  # list of factories/import strings


