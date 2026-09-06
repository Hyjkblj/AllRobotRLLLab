from pathlib import Path
import hashlib
import json
import pytest

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.artifact_service import ArtifactService
from backend.app.application.asset_service import AssetService
from backend.app.application.policy_exporter import ExportResult, file_record, verify_checksums
from backend.app.application.training_service import TrainingService, TrainingServiceError
from backend.app.application.policy_exporter import ExportError
from backend.app.application.reward_catalog import RewardConfigVersionStore, default_reward_config
from backend.app.runtime.contracts import ExternalRunResult
from backend.app.domain.contracts import AssetKind, ExportMetadata, LicenseInfo, TrainingConfig
from backend.app.domain.contracts import Actor
from backend.app.application.run_service import RunService
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.infrastructure.object_store import LocalObjectStore


class StubExporter:
    def export(self, *, output_dir: Path, input_dim: int, output_dim: int, action_scale: float, opset: int = 17) -> ExportResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "policy.pt").write_bytes(b"torchscript-test")
        (output_dir / "policy.onnx").write_bytes(b"onnx-test")
        files = [file_record(output_dir / "policy.pt", root=output_dir, format="torchscript"), file_record(output_dir / "policy.onnx", root=output_dir, format="onnx")]
        metadata = ExportMetadata(policy_input_dim=input_dim, policy_output_dim=output_dim, action_scale=action_scale, onnx_opset=opset, input_name="observation", output_name="action", exporter="stub.v1", runtime="test", smoke_passed=True, files=files)
        return ExportResult(metadata=metadata, files=files, output_dir=output_dir)


def test_training_export_and_sim2sim_smoke_flow(tmp_path: Path) -> None:
    uow = InMemoryUnitOfWork()
    run_service = RunService(uow)
    actor = Actor(user_id="alice")
    project = run_service.create_project(name="P3", actor=actor)
    run, _, _ = run_service.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64)
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path)
    config = TrainingConfig(motion_asset_version_id="motion-1")
    trained = service.train_smoke(run_id=run.run_id, config=config)
    assert trained.checkpoint.sha256
    bundle = service.export(run_id=run.run_id, exporter=StubExporter())
    assert any(file.path.endswith("policy.pt") for file in bundle.files)
    report = service.sim2sim(run_id=run.run_id)
    assert report.status == "PASSED"
    assert service.bundles[run.run_id].status == "READY_TO_DOWNLOAD"


def test_p3_outputs_are_checksum_verified_and_registered(tmp_path: Path) -> None:
    uow = InMemoryUnitOfWork()
    run_service = RunService(uow)
    actor = Actor(user_id="alice")
    project = run_service.create_project(name="P3 artifacts", actor=actor)
    run, _, _ = run_service.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64)
    object_store = LocalObjectStore(tmp_path / "objects")
    artifact_service = ArtifactService(uow, object_store)
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path / "workspace", artifact_service=artifact_service, object_store=object_store)
    config = TrainingConfig(motion_asset_version_id="motion-1")
    service.train_smoke(run_id=run.run_id, config=config)
    service.export(run_id=run.run_id, exporter=StubExporter())
    report = service.sim2sim(run_id=run.run_id)

    bundle = service.bundles[run.run_id]
    bundle_dir = service.bundle_dirs[run.run_id]
    assert verify_checksums(bundle_dir) == []
    assert (bundle_dir / "sim2sim_report.json").is_file()
    archive = bundle_dir.parent / "policy_bundle.tar.gz"
    import tarfile

    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert "policy_bundle/sim2sim_report.json" in names
    artifacts = artifact_service.list_for_run(run_id=run.run_id, actor=actor)
    assert {artifact.kind for artifact in artifacts} == {"checkpoint", "policy_bundle", "sim2sim_report", "policy_bundle_final"}
    assert report.report_sha256 == service.reports[run.run_id].report_sha256
    assert bundle.sim2sim_report is not None
    assert bundle.sim2sim_report.report_sha256 == report.report_sha256
    assert set(bundle.artifact_ids) == {artifact.artifact_id for artifact in artifacts}


def test_p3_state_survives_training_service_recreation(tmp_path: Path) -> None:
    uow = InMemoryUnitOfWork()
    run_service = RunService(uow)
    actor = Actor(user_id="durable-owner")
    project = run_service.create_project(name="P3 durable", actor=actor)
    run, _, _ = run_service.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64)
    object_store = LocalObjectStore(tmp_path / "objects")
    artifact_service = ArtifactService(uow, object_store)
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    first = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path / "workspace", artifact_service=artifact_service, object_store=object_store)
    config = TrainingConfig(motion_asset_version_id="motion-1")
    first.train_smoke(run_id=run.run_id, config=config)

    # A new service instance represents a fresh Celery worker process. It has
    # no in-memory config/checkpoint/export dictionaries to fall back on.
    second = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path / "workspace", artifact_service=artifact_service, object_store=object_store)
    second.export(run_id=run.run_id, exporter=StubExporter())
    third = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path / "workspace", artifact_service=artifact_service, object_store=object_store)
    report = third.sim2sim(run_id=run.run_id)

    assert report.status == "PASSED"
    assert third.bundles[run.run_id].status == "READY_TO_DOWNLOAD"
    assert third.bundles[run.run_id].sim2sim_report is not None
    assert {artifact.kind for artifact in artifact_service.list_for_run(run_id=run.run_id, actor=actor)} == {"checkpoint", "policy_bundle", "sim2sim_report", "policy_bundle_final"}


def test_training_rejects_motion_asset_from_another_project(tmp_path: Path) -> None:
    uow = InMemoryUnitOfWork()
    run_service = RunService(uow)
    actor = Actor(user_id="alice")
    project = run_service.create_project(name="run-project", actor=actor)
    other = run_service.create_project(name="other-project", actor=actor)
    run, _, _ = run_service.create_run(
        actor=actor,
        project_id=project.project_id,
        robot={"robot_id": "unitree_g1_29dof"},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
    )
    assets = AssetService(uow, LocalObjectStore(tmp_path / "objects"))
    _asset, version, _session = assets.create_asset(
        actor=actor,
        project_id=other.project_id,
        kind=AssetKind.MOTION,
        display_name="TrainMotionNPZ",
        original_filename="train_motion.npz",
        license=LicenseInfo(status="declared"),
    )
    assets.mark_validated(asset_version_id=version.asset_version_id, valid=True, sha256="d" * 64, size_bytes=1)
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path / "workspace")
    with pytest.raises(TrainingServiceError) as raised:
        service.prepare_training(run_id=run.run_id, config=TrainingConfig(motion_asset_version_id=version.asset_version_id))
    assert raised.value.code == "TRAIN_MOTION_PROJECT_MISMATCH"


def test_external_export_rejects_logs_without_policy(tmp_path: Path) -> None:
    output_dir = tmp_path / "export"
    output_dir.mkdir()
    (output_dir / "log.txt").write_text("completed", encoding="utf-8")
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({"outputs": [{"path": "log.txt", "sha256": hashlib.sha256(b"completed").hexdigest()}]}), encoding="utf-8")
    execution = ExternalRunResult("export", ("runner",), 0, "", "", output_dir, {"log": output_dir / "log.txt"}, manifest)
    service = object.__new__(TrainingService)
    with pytest.raises(ExportError) as raised:
        service._external_export_result(execution, output_dir=output_dir, input_dim=4, output_dim=2, action_scale=0.25)
    assert raised.value.code == "ISAAC_EXPORT_POLICY_MISSING"


def test_training_preparation_loads_manifest_reward_config_by_hash(tmp_path: Path) -> None:
    uow = InMemoryUnitOfWork()
    run_service = RunService(uow)
    actor = Actor(user_id="reward-owner")
    project = run_service.create_project(name="reward-project", actor=actor)
    reward = default_reward_config(robot_id="unitree_g1_29dof", task_id="g1_mimic")
    reward_store = RewardConfigVersionStore()
    version = reward_store.create(reward, robot_id="unitree_g1_29dof", task_id="g1_mimic")
    run, _, _ = run_service.create_run(
        actor=actor,
        project_id=project.project_id,
        robot={"robot_id": "unitree_g1_29dof"},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256=version.config_sha256,
        training_config_sha256="c" * 64,
    )
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path / "workspace", reward_config_store=reward_store)
    service.prepare_training(run_id=run.run_id, config=TrainingConfig(motion_asset_version_id="motion-1"))
    assert service.reward_configs[run.run_id].model_dump(mode="json") == reward.model_dump(mode="json")
