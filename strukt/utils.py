from __future__ import annotations

import importlib
from typing import Any, Callable, Union


def load_factory(
    factory: Union[str, Callable[..., Any], None],
) -> Callable[..., Any] | None:
    if factory is None:
        return None
    if callable(factory):
        return factory
    if isinstance(factory, str):
        module_path, _, attr = factory.partition(":")
        if not module_path or not attr:
            raise ValueError(
                f"Invalid import string factory '{factory}', expected 'module:attr'"
            )
        module = importlib.import_module(module_path)
        obj = getattr(module, attr)
        if not callable(obj):
            raise TypeError(f"Imported object '{factory}' is not callable")
        return obj
    raise TypeError("Factory must be a callable, import string, or None")


def coerce_factory(factory_like: Any) -> Callable[..., Any] | None:
    """Accept a wide range of inputs and return a factory callable that yields instances.

    Accepted inputs:
    - None -> None
    - Callable -> returned as-is
    - Import string 'module:attr' -> imported object; if callable, returned as-is
    - Class (with __call__ == constructor) -> returns lambda **kw: cls(**kw)
    - Instance (duck-typed by presence of interesting methods like 'handle', 'classify', 'invoke')
      -> returns lambda **kw: instance
    """
    if factory_like is None:
        return None
    # Import string or callable supported directly by load_factory
    if isinstance(factory_like, str) or callable(factory_like):
        try:
            return load_factory(factory_like)  # type: ignore[arg-type]
        except Exception:
            # If callable but not suitable for load_factory, continue below
            if callable(factory_like):
                return factory_like  # type: ignore[return-value]
            raise

    # Class type: instantiate with provided kwargs
    try:
        if hasattr(factory_like, "__mro__") and callable(factory_like):
            cls = factory_like
            return lambda **kwargs: cls(**kwargs)
    except Exception:
        pass

    # Instance: return as-is via nullary factory
    if any(hasattr(factory_like, attr) for attr in ("handle", "classify", "invoke")):
        instance = factory_like
        return lambda **kwargs: instance

    raise TypeError(
        "Unsupported factory type; expected callable, import string, class, or instance"
    )
