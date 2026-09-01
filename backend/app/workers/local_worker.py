"""Single-host worker for Local File Mode.

The worker consumes the durable local queue and invokes the same application
executor used by Celery.  It is intentionally a process boundary: API
restarts do not lose queued envelopes or P3 state.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.artifact_service import ArtifactService
from backend.app.application.run_service import RunService
from backend.app.application.training_service import TrainingService
from backend.app.application.asset_service import AssetService
from backend.app.application.outbox_dispatcher import OutboxDispatcher
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.adapters.motion import MotionSourceRegistry
from backend.app.application.motion_editor import MotionEditor
from backend.app.config.settings import settings
from backend.app.infrastructure.local import build_object_store
from backend.app.infrastructure.local_file import LocalFileUnitOfWork
from backend.app.infrastructure.queue import LocalFileTaskDispatcher
from backend.app.infrastructure.scheduler import LocalRunRecovery
from backend.app.workers.p3_tasks import P3TaskExecutor
from backend.app.runtime.factory import build_runtime_adapters


def build_executor() -> tuple[LocalFileTaskDispatcher, P3TaskExecutor]:
    if settings.storage_mode != "local_file":
        raise RuntimeError("local worker requires ROBOTLAB_MODE=local_file")
    uow = LocalFileUnitOfWork(settings.runtime_root)
    run_service = RunService(uow)
    object_store = build_object_store(settings)
    artifact_service = ArtifactService(uow, object_store)
    adapter = UnitreeG1Adapter(repository_root=settings.repository_root)
    runtime_adapters = build_runtime_adapters(settings, workspace=settings.runtime_root / "external")
    training_runner = runtime_adapters["isaac"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab"} else None
    sim2sim_adapter = runtime_adapters["sim2sim"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "unitree_mujoco"} else None
    training_service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=settings.runtime_root / "runs", artifact_service=artifact_service, object_store=object_store, training_runner=training_runner, sim2sim_adapter=sim2sim_adapter)
    motion_pipeline_service = MotionPipelineService(
        uow=uow,
        object_store=object_store,
        robot_adapter=adapter,
        motion_registry=MotionSourceRegistry(),
        motion_editor=MotionEditor(adapter.get_spec(), ik_solver=adapter.create_ik_solver()),
        asset_service=AssetService(uow, object_store),
        store=MotionPipelineStore(settings.runtime_root / "motion_pipelines"),
        kinematics_compiler=runtime_adapters["compiler"],
        gvhmr_runner=runtime_adapters["gvhmr"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "gmr_gvhmr"} else None,
        gmr_runner=runtime_adapters["gmr"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "gmr_gvhmr"} else None,
    )
    return LocalFileTaskDispatcher(settings.runtime_root / "scheduler"), P3TaskExecutor(training_service, motion_pipeline_service)


def run(*, worker_id: str = "local-worker", poll_interval: float = 1.0, once: bool = False) -> int:
    dispatcher, executor = build_executor()
    outbox = OutboxDispatcher(executor.training_service.run_service.uow, dispatcher)
    dispatcher.recover_running()
    LocalRunRecovery(settings.runtime_root).reconcile(executor.training_service.run_service.uow)
    while True:
        # Move committed API events into the durable queue before claiming a job.
        outbox.dispatch(limit=50)
        envelope = dispatcher.claim(worker_id=worker_id, queues=("cpu", "asset-io", "isaac-gpu", "sim2sim-gpu", "motion-cpu", "motion-gpu", "maintenance"))
        if envelope is None:
            if once:
                return 0
            time.sleep(max(0.1, poll_interval))
            continue
        try:
            payload: dict[str, Any] = {**envelope.payload, "operation": _operation_for_task(envelope.task)}
            _write_process_marker(settings.runtime_root, str(payload["run_id"]))
            executor.execute(payload)
        except Exception:
            dispatcher.fail(envelope.task_id, retry=False)
        else:
            dispatcher.complete(envelope.task_id)
        finally:
            _remove_process_marker(settings.runtime_root, str(envelope.payload.get("run_id", "")))
        if once:
            return 0


def _operation_for_task(task: str) -> str:
    value = task.rsplit(".", 1)[-1].strip().lower()
    if task == "assets.validate" or (value == "validate" and task.startswith("allrobotrl.assets")):
        return "asset_validate"
    if task == "assets.uploading":
        return "asset_uploading"
    if task == "allrobotrl.motion.process" or value == "process":
        return "motion_process"
    if task == "runs.created":
        return "run_validate"
    if task == "runs.retry":
        return "retry"
    if task == "runs.cancelled":
        return "cancelled"
    if value not in {"train", "export", "sim2sim"}:
        raise ValueError(f"unsupported local task: {task}")
    return value


def _write_process_marker(runtime_root, run_id: str) -> None:
    if not run_id:
        return
    directory = runtime_root / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "process.json"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".process.", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"schema_version": "local_process_marker.v1", "pid": os.getpid(), "heartbeat_at": time.time()}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_process_marker(runtime_root, run_id: str) -> None:
    if run_id:
        (runtime_root / "runs" / run_id / "process.json").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotlab-local-worker")
    parser.add_argument("--worker-id", default="local-worker")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    return run(worker_id=args.worker_id, poll_interval=args.poll_interval, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_executor", "main", "run"]
