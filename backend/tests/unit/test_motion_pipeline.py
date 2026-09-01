import numpy as np

from backend.app.adapters.motion import MotionSourceRegistry
from backend.app.application.asset_service import AssetService
from backend.app.application.motion_editor import MotionEditor
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.run_service import RunService
from backend.app.config.settings import settings
from backend.app.domain.contracts import Actor, AssetKind, LicenseInfo
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.infrastructure.object_store import LocalObjectStore
from adapters.unitree_g1_29dof import UnitreeG1Adapter


def test_direct_g1_motion_pipeline_publishes_train_motion(tmp_path):
    uow = InMemoryUnitOfWork()
    object_store = LocalObjectStore(tmp_path / "objects", content_addressed=True)
    run_service = RunService(uow)
    actor = Actor(user_id="motion-owner")
    project = run_service.create_project(name="motion pipeline", actor=actor)
    assets = AssetService(uow, object_store)
    asset, version, session = assets.create_asset(actor=actor, project_id=project.project_id, kind=AssetKind.MOTION, display_name="walk", original_filename="walk.npz", license=LicenseInfo(status="declared"), content_type="application/x-npz")
    source = tmp_path / "walk.npz"
    np.savez(source, joint_pos=np.zeros((20, 29), dtype=np.float32), fps=np.asarray(30.0))
    object_store.put_file(version.object_key, source, content_type="application/x-npz")
    assets.complete_upload(actor=actor, asset_version_id=version.asset_version_id, sha256=None, size_bytes=source.stat().st_size)
    assets.mark_validated(asset_version_id=version.asset_version_id, valid=True, sha256=object_store.stat(version.object_key)["sha256"], size_bytes=source.stat().st_size)
    adapter = UnitreeG1Adapter(repository_root=settings.repository_root)
    pipeline = MotionPipelineService(uow=uow, object_store=object_store, robot_adapter=adapter, motion_registry=MotionSourceRegistry(), motion_editor=MotionEditor(adapter.get_spec()), asset_service=assets, store=MotionPipelineStore(tmp_path / "pipelines"))
    record, submission = pipeline.submit(actor=actor, asset_version_id=version.asset_version_id, sync=True)
    assert submission is None
    assert record.status == "READY"
    assert record.output_asset_version_id
    assert record.train_motion is not None
    assert record.train_motion.arrays["joint_pos"].shape == [20, 29]
    with uow:
        output = uow.assets.version(record.output_asset_version_id)
    assert output is not None and output.status.value == "READY"
    assert object_store.resolve_path(output.object_key).is_file()
