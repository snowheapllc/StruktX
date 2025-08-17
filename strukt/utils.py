from __future__ import annotations

import importlib
from typing import Any, Callable, Union


def load_factory(factory: Union[str, Callable[..., Any], None]) -> Callable[..., Any] | None:
    if factory is None:
        return None
    if callable(factory):
        return factory
    if isinstance(factory, str):
        module_path, _, attr = factory.partition(":")
        if not module_path or not attr:
            raise ValueError(f"Invalid import string factory '{factory}', expected 'module:attr'")
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
        if not callable(obj):
            raise TypeError(f"Imported object '{factory}' is not callable")
        return obj
    raise TypeError("Factory must be a callable, import string, or None")



