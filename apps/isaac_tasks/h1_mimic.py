"""Register the H1 task without importing Isaac Lab in the API process."""

from .delegated import DelegatedIsaacTask


def register_tasks(register_task) -> None:
    register_task("h1_mimic", DelegatedIsaacTask("h1_mimic", "H1_ISAAC_TASK_MODULE"))


__all__ = ["register_tasks"]
