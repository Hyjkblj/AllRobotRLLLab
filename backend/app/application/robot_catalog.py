"""Application-level registry for robot adapters.

The registry owns selection only. Robot facts and conversion rules remain in
the adapter packages, so adding a robot does not require changes to workflow
services or HTTP handlers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class RobotRegistryError(ValueError):
    """Raised when a robot adapter cannot be resolved."""


class RobotAdapterRegistry:
    def __init__(self, adapters: Iterable[Any], *, default_robot_id: str | None = None) -> None:
        values = tuple(adapters)
        if not values:
            raise RobotRegistryError("at least one robot adapter is required")
        self._adapters: dict[str, Any] = {}
        for adapter in values:
            self.register(adapter)
        self.default_robot_id = default_robot_id or next(iter(self._adapters))
        if self.default_robot_id not in self._adapters:
            raise RobotRegistryError(f"default robot adapter is not registered: {self.default_robot_id}")

    def get(self, robot_id: str | None = None) -> Any:
        selected = (robot_id or self.default_robot_id).strip()
        try:
            return self._adapters[selected]
        except KeyError as exc:
            raise RobotRegistryError(f"unknown robot adapter: {selected}") from exc

    def register(self, adapter: Any) -> Any:
        robot_id = str(getattr(adapter, "name", "")).strip()
        if not robot_id:
            raise RobotRegistryError("robot adapter must expose a non-empty name")
        try:
            spec_id = str(adapter.get_spec().robot_id).strip()
        except Exception as exc:
            raise RobotRegistryError(f"robot adapter cannot provide a RobotSpec: {robot_id}") from exc
        if spec_id != robot_id:
            raise RobotRegistryError(f"adapter name does not match RobotSpec.robot_id: {robot_id} != {spec_id}")
        if robot_id in self._adapters:
            raise RobotRegistryError(f"duplicate robot adapter: {robot_id}")
        self._adapters[robot_id] = adapter
        return adapter

    def list(self) -> list[Any]:
        return list(self._adapters.values())

    def specs(self) -> list[Any]:
        return [adapter.get_spec() for adapter in self._adapters.values()]

    def contains(self, robot_id: str) -> bool:
        return robot_id in self._adapters


__all__ = ["RobotAdapterRegistry", "RobotRegistryError"]
