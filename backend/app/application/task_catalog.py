"""Task metadata registry independent from robot and runtime implementations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    robot_id: str
    scene_id: str
    training_provider: str
    sim2sim_adapter: str
    capabilities: tuple[str, ...] = ()


class TaskRegistryError(ValueError):
    pass


class TaskRegistry:
    def __init__(self, tasks: Iterable[TaskSpec] = ()) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        for task in tasks:
            self.register(task)

    def register(self, task: TaskSpec) -> TaskSpec:
        if not task.task_id.strip() or not task.robot_id.strip():
            raise TaskRegistryError("task_id and robot_id are required")
        if not task.scene_id.strip() or not task.training_provider.strip() or not task.sim2sim_adapter.strip():
            raise TaskRegistryError(f"task metadata is incomplete: {task.task_id}")
        if task.task_id in self._tasks:
            raise TaskRegistryError(f"duplicate task: {task.task_id}")
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> TaskSpec:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise TaskRegistryError(f"unknown task: {task_id}") from exc

    def list(self, *, robot_id: str | None = None) -> list[TaskSpec]:
        values = list(self._tasks.values())
        return [task for task in values if robot_id is None or task.robot_id == robot_id]

    def for_robot(self, robot_id: str) -> list[TaskSpec]:
        return self.list(robot_id=robot_id)


def default_task_registry(robot_registry=None, *, legacy_g1: bool = False) -> TaskRegistry:
    """Build task metadata from registered RobotSpec instances.

    A missing registry returns an empty generic catalog. Legacy G1 callers
    must opt in with ``legacy_g1=True`` or ``legacy_g1_task_registry``;
    production composition roots always pass a registry.
    """
    if robot_registry is None and not legacy_g1:
        return TaskRegistry()
    if robot_registry is None:
        return TaskRegistry(
            [
                TaskSpec(
                    task_id="g1_mimic",
                    robot_id="unitree_g1_29dof",
                    scene_id="g1_flat",
                    training_provider="unitree_rl_lab",
                    sim2sim_adapter="unitree_g1_mujoco",
                    capabilities=("mimic",),
                )
            ]
        )
    tasks: list[TaskSpec] = []
    for adapter in robot_registry.list():
        spec = adapter.get_spec()
        for task_id in spec.isaac_task_ids:
            scene_id = spec.default_scene_id or f"{task_id}_scene"
            provider = spec.training_provider or "native_isaac_lab"
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    robot_id=spec.robot_id,
                    scene_id=scene_id,
                    training_provider=provider,
                    sim2sim_adapter=spec.sim2sim_adapter,
                    capabilities=tuple(spec.capabilities),
                )
            )
    return TaskRegistry(tasks)


def legacy_g1_task_registry() -> TaskRegistry:
    """Return the explicit pre-registry G1 compatibility catalog."""

    return default_task_registry(legacy_g1=True)


__all__ = ["TaskRegistry", "TaskRegistryError", "TaskSpec", "default_task_registry", "legacy_g1_task_registry"]
