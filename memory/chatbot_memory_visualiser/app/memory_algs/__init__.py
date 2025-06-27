memory_alg_registry: dict = {}


def register_memory_alg():
    """
    Decorator to populate `alg_registry` using @register_alg()
    """

    def decorator(cls):
        if cls.__name__ in memory_alg_registry:
            raise ValueError(f"Algorithm '{cls.__name__}' is already registered.")
        memory_alg_registry[cls.__name__] = cls
        return cls

    return decorator


# import all algorithm modules to register them
from . import sliding_window

__all__ = [
    "sliding_window",
]
