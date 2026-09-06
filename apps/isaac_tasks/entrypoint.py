"""Stable task entrypoint used by the native Isaac training provider.

Task implementations live in robot/task adapter packages. This module only
defines the process contract and deliberately does not import Isaac Lab at
module import time.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Callable, Protocol


class IsaacTaskEntrypoint(Protocol):
    def train(self, *, task_id: str, manifest: Path, output_dir: Path) -> int: ...

    def play(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int: ...

    def export(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int: ...


_TASKS: dict[str, IsaacTaskEntrypoint] = {}


def register_task(task_id: str, implementation: IsaacTaskEntrypoint) -> None:
    if task_id in _TASKS:
        raise ValueError(f"duplicate Isaac task registration: {task_id}")
    _TASKS[task_id] = implementation


def resolve_task(task_id: str) -> IsaacTaskEntrypoint:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"Isaac task is not registered in the platform: {task_id}") from exc


def load_task_modules(module_names: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Load task modules that register against this stable entrypoint.

    Modules are supplied by the selected Isaac environment through
    ``ISAAC_TASK_MODULES``. Each module may expose ``register_tasks`` and is
    passed this module's ``register_task`` function; no Isaac package is
    imported while the platform API is starting.
    """
    raw = module_names if module_names is not None else tuple(item.strip() for item in os.getenv("ISAAC_TASK_MODULES", "").split(",") if item.strip())
    loaded: list[str] = []
    for name in raw:
        module = importlib.import_module(name)
        register_tasks = getattr(module, "register_tasks", None)
        if not callable(register_tasks):
            raise TypeError(f"Isaac task module must expose register_tasks: {name}")
        register_tasks(register_task)
        loaded.append(name)
    return tuple(loaded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AllRobotRLLLab native Isaac task entrypoint")
    parser.add_argument("operation", choices=("train", "play", "export"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        load_task_modules()
        implementation = resolve_task(args.task)
    except KeyError as exc:
        parser.error(str(exc))
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.operation == "train":
        if not args.manifest:
            parser.error("--manifest is required for train")
        return int(implementation.train(task_id=args.task, manifest=Path(args.manifest).resolve(), output_dir=output_dir))
    if not args.checkpoint:
        parser.error("--checkpoint is required for play/export")
    method: Callable[..., int] = getattr(implementation, args.operation)
    return int(method(task_id=args.task, checkpoint=Path(args.checkpoint).resolve(), output_dir=output_dir))


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    raise SystemExit(main())


__all__ = ["IsaacTaskEntrypoint", "load_task_modules", "main", "register_task", "resolve_task"]
