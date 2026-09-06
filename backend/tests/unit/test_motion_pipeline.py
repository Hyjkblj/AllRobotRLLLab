import numpy as np

from backend.app.adapters.motion import MotionSourceRegistry
from backend.app.application.asset_service import AssetService
from backend.app.application.motion_editor import MotionEditor
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.run_service import RunService
from backend.app.config.settings import settings
from backend.app.domain.contracts import Actor, AssetKind, LicenseInfo
from backend.app.domain.contracts import SourceMotionDescriptor
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.infrastructure.object_store import LocalObjectStore
from adapters.unitree_g1_29dof import UnitreeG1Adapter
from adapters.fixture_biped_2dof import FixtureBipedAdapter
from adapters.unitree_h1_19dof import UnitreeH1Adapter
from backend.app.application.robot_catalog import RobotAdapterRegistry


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


def test_motion_loader_uses_robot_root_height_metadata(tmp_path):
    source = tmp_path / "fixture.npz"
    np.savez(source, joint_pos=np.zeros((2, 2), dtype=np.float32), fps=np.asarray(30.0))
    adapter = FixtureBipedAdapter()
    service = object.__new__(MotionPipelineService)
    service.robot_adapter = adapter
    descriptor = SourceMotionDescriptor(
        asset_version_id="asset-version",
        file_format="npz",
        detected_type="joint_trajectory",
        license=LicenseInfo(status="declared"),
        detector_version="test",
    )
    arrays = service._load_arrays(source, descriptor, adapter=adapter)
    assert arrays.root_pos[:, 2].tolist() == [0.5, 0.5]


def test_motion_pipeline_without_edit_config_uses_registry_default_robot(tmp_path):
    """A legacy adapter argument must not change the registry's target robot."""
    uow = InMemoryUnitOfWork()
    object_store = LocalObjectStore(tmp_path / "objects", content_addressed=True)
    run_service = RunService(uow)
    actor = Actor(user_id="multi-robot-owner")
    project = run_service.create_project(name="multi robot motion", actor=actor)
    assets = AssetService(uow, object_store)
    _asset, version, _session = assets.create_asset(
        actor=actor,
        project_id=project.project_id,
        kind=AssetKind.MOTION,
        display_name="fixture-walk",
        original_filename="fixture.npz",
        license=LicenseInfo(status="declared"),
        content_type="application/x-npz",
    )
    source = tmp_path / "fixture.npz"
    np.savez(source, joint_pos=np.zeros((20, 2), dtype=np.float32), fps=np.asarray(30.0))
    object_store.put_file(version.object_key, source, content_type="application/x-npz")
    assets.complete_upload(actor=actor, asset_version_id=version.asset_version_id, sha256=None, size_bytes=source.stat().st_size)
    assets.mark_validated(asset_version_id=version.asset_version_id, valid=True, sha256=object_store.stat(version.object_key)["sha256"], size_bytes=source.stat().st_size)

    fixture = FixtureBipedAdapter()
    h1 = UnitreeH1Adapter(repository_root=settings.repository_root)
    registry = RobotAdapterRegistry([fixture, h1])
    pipeline = MotionPipelineService(
        uow=uow,
        object_store=object_store,
        # Simulates a compatibility caller still passing a non-default adapter.
        robot_adapter=h1,
        robot_registry=registry,
        motion_registry=MotionSourceRegistry(default_dof=None, default_robot_id=None),
        motion_editor=MotionEditor(h1.get_spec()),
        motion_editors={fixture.name: MotionEditor(fixture.get_spec()), h1.name: MotionEditor(h1.get_spec())},
        asset_service=assets,
        store=MotionPipelineStore(tmp_path / "pipelines"),
    )
    record, submission = pipeline.submit(actor=actor, asset_version_id=version.asset_version_id, sync=True)

    assert submission is None
    assert record.status == "READY"
    assert record.edit_config is not None
    assert record.edit_config.robot_id == fixture.name
    assert record.train_motion is not None
    assert record.train_motion.robot_id == fixture.name
