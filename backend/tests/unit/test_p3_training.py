from pathlib import Path

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.artifact_service import ArtifactService
from backend.app.application.policy_exporter import ExportResult, file_record, verify_checksums
from backend.app.application.training_service import TrainingService
from backend.app.domain.contracts import ExportMetadata, TrainingConfig
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
