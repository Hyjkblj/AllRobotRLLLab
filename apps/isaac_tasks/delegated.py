"""Bridge platform task IDs to implementations inside an Isaac environment."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TaskImplementationError(RuntimeError):
    """Raised when a configured Isaac task implementation is unavailable."""


@dataclass(frozen=True)
class DelegatedIsaacTask:
    task_id: str
    implementation_env: str

    def _implementation(self) -> Any:
        target = os.getenv(self.implementation_env, "").strip()
        if not target:
            raise TaskImplementationError(f"{self.implementation_env} must point to the Isaac task implementation module")
        module = importlib.import_module(target)
        factory = getattr(module, "create_task", None)
        implementation = factory() if callable(factory) else module
        if not any(callable(getattr(implementation, operation, None)) for operation in ("train", "play", "export")):
            raise TaskImplementationError(f"Isaac task module {target} must expose create_task() or train/play/export")
        return implementation

    def _call(self, operation: str, **kwargs: Any) -> int:
        implementation = self._implementation()
        method = getattr(implementation, operation, None)
        if not callable(method):
            raise TaskImplementationError(f"Isaac task {self.task_id} does not implement {operation}")
        value = method(task_id=self.task_id, **kwargs)
        return int(value or 0)

    def train(self, *, task_id: str, manifest: Path, output_dir: Path) -> int:
        return self._call("train", manifest=manifest, output_dir=output_dir)

    def play(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int:
        return self._call("play", checkpoint=checkpoint, output_dir=output_dir)

    def export(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int:
        return self._call("export", checkpoint=checkpoint, output_dir=output_dir)


__all__ = ["DelegatedIsaacTask", "TaskImplementationError"]
