from pathlib import Path
import json

import pytest

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.robot_catalog import RobotAdapterRegistry, RobotRegistryError
from backend.app.application.task_catalog import TaskRegistry, TaskRegistryError, TaskSpec, default_task_registry, legacy_g1_task_registry
from backend.app.domain.contracts import MotionEditConfig, TrainingConfig
from backend.app.application.motion_editor import MotionEditor
from backend.app.domain.motion import MotionArrays
import numpy as np

from adapters.fixture_biped_2dof import FixtureBipedAdapter
from adapters.unitree_h1_19dof import UnitreeH1Adapter
from backend.app.application.run_service import RunService
from backend.app.application.sim2sim_service import FakeSim2SimAdapter
from backend.app.application.training_service import TrainingService
from backend.app.runtime.isaac_runner import IsaacLabRunner
from backend.app.runtime.providers import UnitreeRLLabProvider
from backend.app.application.training_service import TrainingServiceError
from backend.app.domain.state_machine import RunStatus
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.domain.contracts import Actor, CheckpointRecord, P3RunState
from backend.app.application.platform_assembly import build_platform_assembly
from backend.app.config.settings import Settings


def test_robot_registry_resolves_registered_adapter() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    registry = RobotAdapterRegistry([adapter])
    assert registry.get().name == adapter.name
    assert registry.specs()[0].dof == 29


def test_robot_registry_rejects_duplicate_adapter() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    with pytest.raises(RobotRegistryError):
        RobotAdapterRegistry([adapter, adapter])


def test_task_registry_is_robot_scoped_and_config_is_not_literal_bound() -> None:
    assert default_task_registry().list() == []
    registry = legacy_g1_task_registry()
    task = registry.get("g1_mimic")
    assert task.robot_id == "unitree_g1_29dof"
    assert registry.for_robot(task.robot_id) == [task]
    config = TrainingConfig(task_id="future_robot_task", scene_id="future_scene", motion_asset_version_id="motion-1")
    assert config.task_id == "future_robot_task"


def test_task_registry_rejects_duplicates() -> None:
    registry = TaskRegistry([TaskSpec("task-1", "robot-1", "scene", "provider", "sim")])
    with pytest.raises(TaskRegistryError):
        registry.register(TaskSpec("task-1", "robot-2", "scene", "provider", "sim"))


def test_motion_editor_uses_robot_spec_dof_for_a_second_robot_shape() -> None:
    base = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3]).get_spec()
    spec = base.model_copy(
        update={
            "robot_id": "fixture_biped_2dof",
            "model": "Fixture Biped",
            "model_version": "v1",
            "joint_names": base.joint_names[:2],
            "joints": base.joints[:2],
            "dof": 2,
            "actuation": base.actuation.model_copy(update={"torque_limits": base.actuation.torque_limits[:2]}),
            "sim2sim_adapter": "fixture_sim",
            "isaac_task_ids": ["fixture_mimic"],
            "default_scene_id": "fixture_flat",
        }
    )
    arrays = MotionArrays(30.0, np.zeros((3, 2)), np.zeros((3, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)), tuple(spec.joint_names), "xyzw", "world_z_up")
    config = MotionEditConfig(source_motion_version_id="m", robot_id=spec.robot_id)
    result = MotionEditor(spec).apply(arrays, config)
    assert result.validation.valid


def test_training_service_resolves_second_robot_from_registry(tmp_path: Path) -> None:
    fixture = FixtureBipedAdapter()
    registry = RobotAdapterRegistry([fixture])
    tasks = default_task_registry(registry)
    uow = InMemoryUnitOfWork()
    runs = RunService(uow)
    actor = Actor(user_id="fixture-owner")
    project = runs.create_project(name="fixture", actor=actor)
    run, _, _ = runs.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": fixture.name}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64)
    service = TrainingService(run_service=runs, robot_registry=registry, task_registry=tasks, robot_adapter=fixture, workspace=tmp_path, sim2sim_adapter=FakeSim2SimAdapter(name="fixture_biped_sim2sim"))
    config = TrainingConfig(task_id="fixture_mimic", scene_id="fixture_flat", motion_asset_version_id="motion-1")
    result = service.train_smoke(run_id=run.run_id, config=config)
    payload = json.loads(Path(result.checkpoint.uri).read_text(encoding="utf-8"))
    assert payload["robot_id"] == fixture.name
    assert payload["action_dim"] == fixture.get_spec().dof


def test_h1_adapter_validates_public_mujoco_model_and_derives_task() -> None:
    adapter = UnitreeH1Adapter(repository_root=Path(__file__).resolve().parents[3])
    spec = adapter.get_spec()
    assert spec.dof == 19
    assert adapter.self_check().valid
    registry = RobotAdapterRegistry([adapter])
    task = default_task_registry(registry).get("h1_mimic")
    assert task.robot_id == adapter.name
    assert task.scene_id == "h1_flat"


def test_platform_assembly_registers_multiple_robots_and_editors(monkeypatch, tmp_path: Path) -> None:
    settings = Settings()
    monkeypatch.setattr(settings, "robot_adapter_modules", ("adapters.unitree_g1_29dof", "adapters.unitree_h1_19dof"))
    assembly = build_platform_assembly(settings, workspace=tmp_path / "external")

    assert {adapter.name for adapter in assembly.robot_registry.list()} == {"unitree_g1_29dof", "unitree_h1_19dof"}
    assert set(assembly.motion_editors) == {"unitree_g1_29dof", "unitree_h1_19dof"}
    assert {task.task_id for task in assembly.task_registry.list()} == {"g1_mimic", "h1_mimic"}
    # H1 has no vendor-specific IK solver; composition must still expose a
    # usable editor instead of assuming every adapter implements one.
    assert assembly.motion_editors["unitree_h1_19dof"].ik_solver is None
    assert assembly.runtime_adapters["sim2sim_adapters"]["mujoco_h1"].command_env == "H1_SIM2SIM_COMMAND"


def test_training_service_rejects_g1_provider_for_h1_run(tmp_path: Path) -> None:
    adapter = UnitreeH1Adapter(repository_root=Path(__file__).resolve().parents[3])
    registry = RobotAdapterRegistry([adapter])
    tasks = default_task_registry(registry)
    runs = RunService(InMemoryUnitOfWork())
    actor = Actor(user_id="h1-owner")
    project = runs.create_project(name="h1", actor=actor)
    run, _, _ = runs.create_run(
        actor=actor,
        project_id=project.project_id,
        robot={"robot_id": adapter.name},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
    )
    provider = UnitreeRLLabProvider(IsaacLabRunner.__new__(IsaacLabRunner))
    service = TrainingService(
        run_service=runs,
        robot_adapter=adapter,
        robot_registry=registry,
        task_registry=tasks,
        training_runner=provider,
        workspace=tmp_path,
    )
    config = TrainingConfig(task_id="h1_mimic", scene_id="h1_flat", motion_asset_version_id="motion-1")
    with pytest.raises(TrainingServiceError, match="does not support") as raised:
        service.train(run_id=run.run_id, config=config)
    assert raised.value.code == "TRAINING_PROVIDER_ROBOT_UNSUPPORTED"


def test_training_service_derives_h1_task_and_scene_when_omitted(tmp_path: Path) -> None:
    adapter = UnitreeH1Adapter(repository_root=Path(__file__).resolve().parents[3])
    registry = RobotAdapterRegistry([adapter])
    runs = RunService(InMemoryUnitOfWork())
    actor = Actor(user_id="h1-smoke-owner")
    project = runs.create_project(name="h1-smoke", actor=actor)
    run, _, _ = runs.create_run(
        actor=actor,
        project_id=project.project_id,
        robot={"robot_id": adapter.name},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
    )
    service = TrainingService(run_service=runs, robot_adapter=adapter, robot_registry=registry, workspace=tmp_path)
    result = service.train(run_id=run.run_id, config=TrainingConfig(motion_asset_version_id="motion-1"))
    payload = json.loads(Path(result.checkpoint.uri).read_text(encoding="utf-8"))
    assert payload["task_id"] == "h1_mimic"
    assert service.configs[run.run_id].scene_id == "h1_flat"


def test_sim2sim_does_not_fallback_to_fake_for_real_provider(tmp_path: Path, monkeypatch) -> None:
    adapter = UnitreeH1Adapter(repository_root=Path(__file__).resolve().parents[3])
    registry = RobotAdapterRegistry([adapter])
    runs = RunService(InMemoryUnitOfWork())
    actor = Actor(user_id="h1-sim-owner")
    project = runs.create_project(name="h1-sim", actor=actor)
    run, _, _ = runs.create_run(
        actor=actor,
        project_id=project.project_id,
        robot={"robot_id": adapter.name},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
    )
    class UnrelatedSimAdapter:
        name = "fixture_sim2sim"

    service = TrainingService(run_service=runs, robot_adapter=adapter, robot_registry=registry, workspace=tmp_path, training_runner=None, sim2sim_adapter=UnrelatedSimAdapter())
    assert not service.has_sim2sim_adapter_for_robot(adapter.name)
    monkeypatch.setattr("backend.app.application.training_service.settings", type("Deployment", (), {"is_deployed": False})())
    # Move the run to the precondition expected by sim2sim; no adapter should
    # still result in a clear failure instead of a fake report.
    for status in (RunStatus.UPLOADING, RunStatus.UPLOADED, RunStatus.VALIDATING, RunStatus.MOTION_COMPILING, RunStatus.MOTION_READY, RunStatus.TRAINING_PREPARING, RunStatus.TRAINING, RunStatus.TRAINING_SUCCEEDED, RunStatus.EXPORTING, RunStatus.EXPORTED):
        if runs.get_run(run_id=run.run_id, actor=actor)[0].status != status:
            runs.transition_run(run_id=run.run_id, target=status, stage="test", message="test")
    service._persist_state(P3RunState(
        run_id=run.run_id,
        attempt_id=run.current_attempt_id,
        training_config=TrainingConfig(task_id="h1_mimic", scene_id="h1_flat", motion_asset_version_id="motion-1"),
        checkpoint=CheckpointRecord(checkpoint_id="checkpoint", run_id=run.run_id, attempt_id=run.current_attempt_id, uri=str(tmp_path / "checkpoint.pt"), sha256="c" * 64, iteration=1, created_at="2026-01-01T00:00:00Z"),
        updated_at="2026-01-01T00:00:00Z",
    ))
    with pytest.raises(TrainingServiceError, match="no sim2sim adapter") as raised:
        service.sim2sim(run_id=run.run_id)
    assert raised.value.code == "SIM2SIM_ADAPTER_UNSUPPORTED"
