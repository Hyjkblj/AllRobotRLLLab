from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.application.asset_service import AssetService
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.run_service import RunService, RunServiceError
from backend.app.domain.contracts import Actor, AssetKind, LicenseInfo
from backend.app.infrastructure.local_file import LocalFileUnitOfWork
from backend.app.infrastructure.object_store import LocalObjectStore


def _setup(tmp_path: Path):
    root = tmp_path / "runtime"
    uow = LocalFileUnitOfWork(root)
    store = LocalObjectStore(tmp_path / "objects", content_addressed=True)
    actor = Actor(user_id="owner")
    project = RunService(uow).create_project(name="test deletion", actor=actor)
    return root, uow, store, actor, project


def _asset(uow, store, actor, project_id, kind, filename, tmp_path):
    asset, version, _ = AssetService(uow, store).create_asset(
        actor=actor, project_id=project_id, kind=kind, display_name=filename,
        original_filename=filename, license=LicenseInfo(status="declared"),
    )
    source = tmp_path / filename
    source.write_bytes(b"uploaded content")
    store.put_file(version.object_key, source)
    return asset, version


def test_project_delete_removes_uploads_and_durable_metadata(tmp_path):
    root, uow, store, actor, project = _setup(tmp_path)
    versions = [
        _asset(uow, store, actor, project.project_id, kind, name, tmp_path)[1]
        for kind, name in ((AssetKind.VIDEO, "clip.mp4"), (AssetKind.MOTION, "walk.npz"))
    ]
    RunService(uow).delete_project(project_id=project.project_id, actor=actor, object_store=store, pipeline_store=MotionPipelineStore(tmp_path / "pipelines"))

    assert all(not store.resolve_path(version.object_key).exists() for version in versions)
    restored = LocalFileUnitOfWork(root)
    with restored:
        assert restored.projects.get(project.project_id) is None
        assert restored.assets.list_for_project(project.project_id) == []
        assert all(restored.assets.version(version.asset_version_id) is None for version in versions)


def test_delete_non_video_asset_removes_object_and_metadata(tmp_path):
    root, uow, store, actor, project = _setup(tmp_path)
    asset, version = _asset(uow, store, actor, project.project_id, AssetKind.MOTION, "walk.npz", tmp_path)
    AssetService(uow, store).delete_asset_version(actor=actor, asset_version_id=version.asset_version_id, pipeline_store=MotionPipelineStore(tmp_path / "pipelines"))

    assert not store.resolve_path(version.object_key).exists()
    with LocalFileUnitOfWork(root) as restored:
        assert restored.assets.version(version.asset_version_id) is None
        assert restored.assets.get(asset.asset_id) is None


def test_video_validation_moves_from_validating_to_ready(tmp_path):
    root, uow, store, actor, project = _setup(tmp_path)
    _, version = _asset(uow, store, actor, project.project_id, AssetKind.VIDEO, "clip.mp4", tmp_path)
    AssetService(uow, store).complete_upload(actor=actor, asset_version_id=version.asset_version_id, sha256=None, size_bytes=None)
    pipeline = object.__new__(MotionPipelineService)
    pipeline.uow = uow
    pipeline.object_store = store
    pipeline.asset_service = AssetService(uow, store)

    assert pipeline.validate_asset_version(version.asset_version_id).status.value == "READY"
    with LocalFileUnitOfWork(root) as restored:
        assert restored.assets.version(version.asset_version_id).status.value == "READY"


def test_delete_refuses_pipeline_source_and_output(tmp_path):
    _, uow, store, actor, project = _setup(tmp_path)
    _, source = _asset(uow, store, actor, project.project_id, AssetKind.VIDEO, "clip.mp4", tmp_path)
    _, output = _asset(uow, store, actor, project.project_id, AssetKind.MOTION, "result.npz", tmp_path)
    record = SimpleNamespace(output_asset_version_id=output.asset_version_id)
    pipelines = SimpleNamespace(list_for_source=lambda version_id: [record] if version_id == source.asset_version_id else [])

    for version in (source, output):
        with pytest.raises(RunServiceError) as error:
            AssetService(uow, store).delete_asset_version(actor=actor, asset_version_id=version.asset_version_id, pipeline_store=pipelines)
        assert error.value.code == "ASSET_IN_USE"
        assert store.resolve_path(version.object_key).exists()

    with pytest.raises(RunServiceError) as error:
        RunService(uow).delete_project(project_id=project.project_id, actor=actor, object_store=store, pipeline_store=pipelines)
    assert error.value.code == "PROJECT_HAS_PIPELINES"


def test_project_delete_refuses_runs_and_preserves_upload(tmp_path):
    _, uow, store, actor, project = _setup(tmp_path)
    _, version = _asset(uow, store, actor, project.project_id, AssetKind.MOTION, "walk.npz", tmp_path)
    RunService(uow).create_run(
        actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"},
        motion={"motion_asset_version_id": version.asset_version_id},
        reward_config_sha256="a" * 64, training_config_sha256="b" * 64,
    )
    with pytest.raises(RunServiceError) as error:
        RunService(uow).delete_project(project_id=project.project_id, actor=actor, object_store=store)
    assert error.value.code == "PROJECT_HAS_RUNS"
    assert store.resolve_path(version.object_key).exists()


def test_non_owner_cannot_delete_project(tmp_path):
    _, uow, store, _actor, project = _setup(tmp_path)
    with pytest.raises(RunServiceError) as error:
        RunService(uow).delete_project(project_id=project.project_id, actor=Actor(user_id="stranger"), object_store=store)
    assert error.value.code == "PROJECT_ACCESS_DENIED"
