"""Dependency-light launcher for the platform-owned G1 Isaac task.

This module intentionally imports no Isaac packages at module import time.
``AppLauncher`` must create the Omniverse application before the runtime task
module imports Isaac Lab, PhysX or RSL-RL.
"""

from __future__ import annotations

import os
import importlib
from pathlib import Path


class G1NativeIsaacTask:
    """Run one native G1 operation in the current short-lived child process."""

    @staticmethod
    def _launch(operation: str, callback, **kwargs) -> int:
        from isaaclab.app import AppLauncher

        device = os.getenv("ISAAC_DEVICE", "cuda:0").strip() or "cuda:0"
        launcher = AppLauncher(
            {
                "headless": True,
                "device": device,
                "enable_cameras": operation == "play" and os.getenv("G1_PLAY_VIDEO", "0") == "1",
            }
        )
        application = launcher.app
        try:
            # Isaac imports are delayed until SimulationApp exists.
            g1_runtime = importlib.import_module("apps.isaac_tasks.g1_runtime")

            return int(callback(g1_runtime, device=device, **kwargs) or 0)
        finally:
            application.close()

    def train(self, *, task_id: str, manifest: Path, output_dir: Path) -> int:
        return self._launch(
            "train",
            lambda runtime, **values: runtime.train(task_id=task_id, manifest=manifest, output_dir=output_dir, **values),
        )

    def export(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int:
        return self._launch(
            "export",
            lambda runtime, **values: runtime.export(task_id=task_id, checkpoint=checkpoint, output_dir=output_dir, **values),
        )

    def play(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int:
        return self._launch(
            "play",
            lambda runtime, **values: runtime.play(task_id=task_id, checkpoint=checkpoint, output_dir=output_dir, **values),
        )


def create_task() -> G1NativeIsaacTask:
    return G1NativeIsaacTask()


__all__ = ["G1NativeIsaacTask", "create_task"]
