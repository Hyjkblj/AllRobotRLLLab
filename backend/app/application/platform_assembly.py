"""Composition root for platform adapters and runtime ports.

Only this module knows how concrete robot modules are loaded.  HTTP handlers
and workers consume the resulting registries and dictionaries, keeping their
business logic independent of a vendor model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.app.application.motion_editor import MotionEditor
from backend.app.application.robot_loader import load_robot_adapters
from backend.app.application.task_catalog import TaskRegistry, default_task_registry
from backend.app.config.settings import Settings
from backend.app.runtime.factory import build_runtime_adapters


@dataclass(frozen=True)
class PlatformAssembly:
    robot_registry: object
    task_registry: TaskRegistry
    runtime_adapters: dict[str, object]
    motion_editors: dict[str, MotionEditor]

    @property
    def default_adapter(self):
        return self.robot_registry.get()


def build_platform_assembly(settings: Settings, *, workspace: Path) -> PlatformAssembly:
    robot_registry = load_robot_adapters(repository_root=settings.repository_root, modules=settings.robot_adapter_modules)
    task_registry = default_task_registry(robot_registry)
    runtime_adapters = build_runtime_adapters(settings, workspace=workspace, robot_registry=robot_registry)
    editors: dict[str, MotionEditor] = {}
    for adapter in robot_registry.list():
        solver_factory = getattr(adapter, "create_ik_solver", None)
        solver = solver_factory() if callable(solver_factory) else None
        editors[adapter.name] = MotionEditor(adapter.get_spec(), ik_solver=solver)
    return PlatformAssembly(robot_registry, task_registry, runtime_adapters, editors)


__all__ = ["PlatformAssembly", "build_platform_assembly"]
