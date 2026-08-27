import json
from pathlib import Path

from backend.app.application.run_service import RunService
from backend.app.domain.contracts import Actor
from backend.app.domain.state_machine import RunStatus
from backend.app.infrastructure.local_file import LocalFileUnitOfWork
from backend.app.infrastructure.object_store import LocalObjectStore
from backend.app.infrastructure.queue import LocalFileTaskDispatcher
from backend.app.infrastructure.scheduler import GpuSnapshot, LocalGpuScheduler, LocalRunRecovery
from backend.app.infrastructure.robot_registry import LocalRobotRegistry, RobotRegistryError


def _create_run(root: Path) -> tuple[str, str]:
    uow = LocalFileUnitOfWork(root)
    service = RunService(uow)
    actor = Actor(user_id="local-owner")
    project = service.create_project(name="durable project", actor=actor)
    run, attempt, duplicate = service.create_run(
        actor=actor,
        project_id=project.project_id,
        robot={"robot_id": "unitree_g1_29dof"},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
        idempotency_key="local-run-1",
    )
    assert not duplicate
    service.append_event(run_id=run.run_id, event_type="metric", stage="training", message="return", payload={"value": 1.0})
    return project.project_id, run.run_id


def test_local_file_uow_survives_process_recreation(tmp_path: Path) -> None:
    project_id, run_id = _create_run(tmp_path / "runtime")
    runtime = tmp_path / "runtime"
    run_dir = runtime / "runs" / run_id
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "state.json").is_file()
    assert (run_dir / "events.jsonl").read_text(encoding="utf-8").count("\n") == 2
    assert (run_dir / "metrics.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (runtime / "index.json").is_file()

    restored = LocalFileUnitOfWork(runtime)
    service = RunService(restored)
    run, attempts = service.get_run(run_id=run_id, actor=Actor(user_id="local-owner"))
    assert run.project_id == project_id
    assert len(attempts) == 1
    assert len(service.list_events(run_id=run_id, actor=Actor(user_id="local-owner"))) == 2
    duplicate, attempt, was_duplicate = service.create_run(
        actor=Actor(user_id="local-owner"),
        project_id=project_id,
        robot={"robot_id": "unitree_g1_29dof"},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
        idempotency_key="local-run-1",
    )
    assert was_duplicate
    assert duplicate.run_id == run_id
    assert attempt.attempt_id == attempts[0].attempt_id


def test_local_queue_is_durable_and_idempotent(tmp_path: Path) -> None:
    queue = LocalFileTaskDispatcher(tmp_path / "scheduler")
    first = queue.enqueue(queue="isaac-gpu", task="train", payload={"run_id": "r"}, idempotency_key="k")
    assert queue.enqueue(queue="isaac-gpu", task="train", payload={"run_id": "r"}, idempotency_key="k") == first
    assert queue.pending()[0].task_id == first
    claimed = queue.claim(worker_id="worker-1", queues=("isaac-gpu",))
    assert claimed is not None and claimed.task_id == first
    assert queue.pending() == []
    assert queue.complete(first)
    assert LocalFileTaskDispatcher(tmp_path / "scheduler").pending() == []


def test_local_gpu_scheduler_honors_exclusive_and_memory_limits(tmp_path: Path) -> None:
    scheduler = LocalGpuScheduler(tmp_path / "scheduler", max_jobs_per_gpu=1)
    snapshot = GpuSnapshot(index=0, uuid="GPU-0", name="RTX 4060", memory_total_mb=8192, memory_used_mb=0, utilization_percent=0)
    lease = scheduler.acquire(run_id="run-1", owner_id="owner", memory_gb=2, snapshots=[snapshot])
    assert lease is not None
    assert scheduler.acquire(run_id="run-2", owner_id="owner", memory_gb=2, snapshots=[snapshot]) is None
    assert scheduler.release(lease.lease_id)
    assert scheduler.acquire(run_id="run-3", owner_id="owner", memory_gb=2, exclusive=True, snapshots=[snapshot]) is not None


def test_content_addressed_local_objects_keep_api_keys(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"content")
    store = LocalObjectStore(tmp_path / "artifacts", content_addressed=True)
    metadata = store.put_file("projects/p/file.bin", source)
    digest = str(metadata["sha256"])
    assert (tmp_path / "artifacts" / "sha256" / digest / "projects" / "p" / "file.bin").is_file()
    assert store.stat("projects/p/file.bin")["sha256"] == digest
    assert json.loads((tmp_path / "artifacts" / "index.json").read_text(encoding="utf-8"))["projects/p/file.bin"]


def test_robot_registry_records_external_package_without_copying(tmp_path: Path) -> None:
    package = tmp_path / "robot"
    package.mkdir()
    (package / "meshes").mkdir()
    (package / "meshes" / "base.stl").write_bytes(b"mesh")
    (package / "robot.urdf").write_text('<robot name="demo"><link name="base"><visual><geometry><mesh filename="meshes/base.stl"/></geometry></visual></link></robot>', encoding="utf-8")
    from adapters.unitree_g1_29dof import UnitreeG1Adapter

    spec = UnitreeG1Adapter().get_spec().model_dump(mode="json")
    spec["robot_id"] = "demo"
    (package / "robot_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    registry = LocalRobotRegistry(tmp_path / "runtime")
    record = registry.add(package)
    assert record["robot_id"] == "demo"
    assert record["verification"] == "verified"
    assert len(registry.list()) == 1
    assert not (tmp_path / "runtime" / "robots" / "demo").exists()


def test_robot_registry_rejects_missing_mesh_reference(tmp_path: Path) -> None:
    package = tmp_path / "robot"
    package.mkdir()
    (package / "robot.urdf").write_text('<robot name="demo"><link name="base"><visual><geometry><mesh filename="missing.stl"/></geometry></visual></link></robot>', encoding="utf-8")
    try:
        LocalRobotRegistry(tmp_path / "runtime").add(package)
    except RobotRegistryError as exc:
        assert "missing asset reference" in str(exc)
    else:
        raise AssertionError("missing mesh reference was accepted")


def test_local_run_recovery_marks_lost_worker(tmp_path: Path) -> None:
    project_id, run_id = _create_run(tmp_path / "runtime")
    uow = LocalFileUnitOfWork(tmp_path / "runtime")
    with uow:
        run = uow.runs.get(run_id)
        attempt = uow.runs.attempt(run.current_attempt_id)
        uow.runs.update(run.model_copy(update={"status": RunStatus.TRAINING}))
        uow.runs.update_attempt(attempt.model_copy(update={"status": RunStatus.TRAINING, "last_heartbeat_at": "2020-01-01T00:00:00+00:00"}))
    recovered = LocalRunRecovery(tmp_path / "runtime", stale_after_seconds=1).reconcile(LocalFileUnitOfWork(tmp_path / "runtime"), now=2000000000)
    assert recovered == [run_id]
    restored = LocalFileUnitOfWork(tmp_path / "runtime")
    with restored:
        assert restored.runs.get(run_id).status == RunStatus.INTERRUPTED
