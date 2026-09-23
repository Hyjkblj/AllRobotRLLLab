"""Platform-owned Isaac task registration boundary."""

from importlib import import_module
from typing import Any

__all__ = ["IsaacTaskEntrypoint", "load_task_modules", "register_task", "resolve_task"]


def __getattr__(name: str) -> Any:
    """Keep package imports lazy so ``python -m ...entrypoint`` is warning-free."""

    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module("apps.isaac_tasks.entrypoint"), name)
