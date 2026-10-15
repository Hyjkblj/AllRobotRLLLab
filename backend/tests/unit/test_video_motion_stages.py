from pathlib import Path

import numpy as np

from adapters.fixture_biped_2dof import FixtureBipedAdapter
from backend.app.adapters.motion import MotionSourceRegistry
from backend.app.application.asset_service import AssetService
from backend.app.application.motion_editor import MotionEditor
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.run_service import RunService
from backend.app.domain.contracts import Actor, AssetKind, LicenseInfo
from backend.app.infrastructure.local_file import LocalFileUnitOfWork
from backend.app.infrastructure.object_store import LocalObjectStore
from backend.app.infrastructure.queue import LocalFileTaskDispatcher
from backend.app.runtime.contracts import RunnerError


class FakeGVHMR:
    def run(self, *, video_path, output_dir):
        assert video_path.is_file()
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "hmr4d_results.pt"
        output.write_bytes(b"human motion")
        return output, None

    def validate_output(self, output):
        assert output.read_bytes() == b"human motion"


class FakeGMR:
    def __init__(self):
        self.fail_once = True

    def run(self, *, source_path, output_dir, robot, target_dof):
        if self.fail_once:
            self.fail_once = False
            raise RunnerError("GMR_TEST_FAILURE", "temporary GMR failure")
        assert source_path.is_file()
        assert robot == "fixture_biped"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "retarget_motion.npz"
        np.savez(output, joint_pos=np.zeros((20, target_dof), dtype=np.float32), fps=np.asarray(30.0))
        return output, None


def test_video_validation_and_separate_motion_stages(tmp_path: Path):
    runtime = tmp_path / "runtime"
    uow = LocalFileUnitOfWork(runtime)
    object_store = LocalObjectStore(tmp_path / "objects")
    queue = LocalFileTaskDispatcher(runtime / "scheduler")
    actor = Actor(user_id="owner")
    project = RunService(uow).create_project(name="video workflow", actor=actor)
    asset_service = AssetService(uow, object_store)
    _asset, version, _session = asset_service.create_asset(
        actor=actor, project_id=project.project_id, kind=AssetKind.VIDEO,
        display_name="clip", original_filename="clip.mp4", license=LicenseInfo(status="declared"),
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video bytes")
    object_store.put_file(version.object_key, source)
    assert asset_service.complete_upload(actor=actor, asset_version_id=version.asset_version_id, sha256=None, size_bytes=None).status.value == "VALIDATING"

    adapter = FixtureBipedAdapter()
    service = MotionPipelineService(
        uow=uow, object_store=object_store, robot_adapter=adapter,
        motion_registry=MotionSourceRegistry(), motion_editor=MotionEditor(adapter.get_spec()),
        asset_service=asset_service, store=MotionPipelineStore(runtime / "motion_pipelines"),
        task_dispatcher=queue, gvhmr_runner=FakeGVHMR(), gmr_runner=FakeGMR(),
    )
    assert service.validate_asset_version(version.asset_version_id).status.value == "READY"
    record, submission = service.submit(actor=actor, asset_version_id=version.asset_version_id, phase="gvhmr")
    assert submission.queue == "motion-gpu"
    assert queue.pending("motion-gpu")[0].task == "allrobotrl.motion.gvhmr"
    assert service.process_gvhmr(record.pipeline_id).status == "GVHMR_READY"

    queued, submission = service.enqueue_phase(actor=actor, pipeline_id=record.pipeline_id, phase="gmr")
    assert queued.status == "QUEUED" and submission.operation == "motion_gmr"
    assert service.process_gmr(record.pipeline_id).status == "FAILED"
    failed_task_id = submission.task_id
    queued, submission = service.enqueue_phase(actor=actor, pipeline_id=record.pipeline_id, phase="gmr")
    assert queued.status == "QUEUED" and submission.task_id != failed_task_id
    assert service.process_gmr(record.pipeline_id).status == "GMR_READY"

    queued, submission = service.enqueue_phase(actor=actor, pipeline_id=record.pipeline_id, phase="compile")
    assert queued.status == "QUEUED" and submission.operation == "motion_compile"
    completed = service.process_compile(record.pipeline_id)
    assert completed.status == "READY", (completed.error_code, completed.error_message)
    assert completed.output_asset_version_id
    assert LocalFileUnitOfWork(runtime).assets.version(completed.output_asset_version_id).status.value == "READY"
