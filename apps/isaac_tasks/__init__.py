"""Platform-owned Isaac task registration boundary."""

from .entrypoint import IsaacTaskEntrypoint, load_task_modules, register_task, resolve_task

__all__ = ["IsaacTaskEntrypoint", "load_task_modules", "register_task", "resolve_task"]
