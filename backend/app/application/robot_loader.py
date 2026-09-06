"""Load robot adapters at the composition root.

Application services receive a registry and never import a concrete vendor
adapter.  Deployments can extend ``ROBOT_ADAPTER_MODULES`` with comma-separated
modules exposing ``create_adapter(repository_root=...)``.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from backend.app.application.robot_catalog import RobotAdapterRegistry, RobotRegistryError


def load_robot_adapters(*, repository_root: Path, modules: Iterable[str]) -> RobotAdapterRegistry:
    adapters: list[Any] = []
    for raw_name in modules:
        module_name = str(raw_name).strip()
        if not module_name:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RobotRegistryError(f"unable to import robot adapter module: {module_name}") from exc
        factory = getattr(module, "create_adapter", None)
        if not callable(factory):
            raise RobotRegistryError(f"robot adapter module must expose create_adapter: {module_name}")
        try:
            adapter = factory(repository_root=repository_root)
        except TypeError:
            adapter = factory()
        adapters.append(adapter)
    if not adapters:
        raise RobotRegistryError("ROBOT_ADAPTER_MODULES must contain at least one adapter module")
    return RobotAdapterRegistry(adapters)


__all__ = ["load_robot_adapters"]
