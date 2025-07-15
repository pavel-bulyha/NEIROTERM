# utils/registry.py

from typing import Callable, Dict

# Dictionary: model name → run function
_model_registry: Dict[str, Callable] = {}

def register_model(name: str):
    """
    Decorator for registering a model entry point.

    Usage:
        @register_model("Perceptron")
        def PerceptronRUN(...):
            ...
    """
    def decorator(fn: Callable):
        if name in _model_registry:
            raise KeyError(f"Model '{name}' already registered")
        _model_registry[name] = fn
        return fn
    return decorator

def get_model(name: str) -> Callable:
    """
    Returns the registered function by name.
    """
    try:
        return _model_registry[name]
    except KeyError:
        raise KeyError(f"Model '{name}' is not registered") from None

def list_models() -> list:
    """
    List of all registered names.
    """
    return list(_model_registry.keys())
