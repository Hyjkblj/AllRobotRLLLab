"""Register the G1 task without importing Isaac Lab in the API process."""

from .delegated import DelegatedIsaacTask


def register_tasks(register_task) -> None:
    register_task(
        "g1_mimic",
        DelegatedIsaacTask(
            "g1_mimic",
            "G1_ISAAC_TASK_MODULE",
            default_module="apps.isaac_tasks.g1_native",
        ),
    )


__all__ = ["register_tasks"]
