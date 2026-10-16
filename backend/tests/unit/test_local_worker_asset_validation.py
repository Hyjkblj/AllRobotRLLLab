from types import SimpleNamespace

from backend.app.infrastructure.queue import TaskEnvelope
from backend.app.workers import local_worker


def test_local_worker_maps_video_full_motion_task_to_motion_process():
    assert local_worker._operation_for_task("allrobotrl.motion.process_gpu") == "motion_process"


def test_local_worker_executes_asset_validation_without_run_id(tmp_path, monkeypatch):
    envelope = TaskEnvelope(
        task_id="task-1",
        queue="asset-io",
        task="allrobotrl.assets.validate",
        payload={"asset_version_id": "version-1"},
        idempotency_key="validation-1",
    )
    executed = []
    completed = []
    failed = []

    dispatcher = SimpleNamespace(
        recover_running=lambda: 0,
        recover_failed=lambda **_kwargs: 0,
        claim=lambda **_kwargs: envelope,
        complete=lambda task_id: completed.append(task_id),
        fail=lambda task_id, **_kwargs: failed.append(task_id),
    )
    executor = SimpleNamespace(
        training_service=SimpleNamespace(run_service=SimpleNamespace(uow=object())),
        execute=lambda payload: executed.append(payload),
    )
    monkeypatch.setattr(local_worker, "build_executor", lambda: (dispatcher, executor))
    monkeypatch.setattr(local_worker, "OutboxDispatcher", lambda *_args: SimpleNamespace(dispatch=lambda **_kwargs: 0))
    monkeypatch.setattr(local_worker, "LocalRunRecovery", lambda *_args: SimpleNamespace(reconcile=lambda *_args: None))
    monkeypatch.setattr(local_worker.settings, "runtime_root", tmp_path)

    assert local_worker.run(once=True) == 0
    assert executed == [{"asset_version_id": "version-1", "operation": "asset_validate"}]
    assert completed == ["task-1"]
    assert failed == []
    assert not (tmp_path / "runs").exists()


def test_local_worker_marks_failed_validation_rejected(tmp_path, monkeypatch, capsys):
    envelope = TaskEnvelope(
        task_id="task-2", queue="asset-io", task="allrobotrl.assets.validate",
        payload={"asset_version_id": "version-2"}, idempotency_key="validation-2",
    )
    rejected = []
    failed = []
    dispatcher = SimpleNamespace(
        recover_running=lambda: 0, recover_failed=lambda **_kwargs: 0,
        claim=lambda **_kwargs: envelope, complete=lambda *_args: None,
        fail=lambda task_id, **_kwargs: failed.append(task_id),
    )
    class UnitOfWork:
        assets = SimpleNamespace(version=lambda _version_id: SimpleNamespace(status=SimpleNamespace(value="VALIDATING")))
        def __enter__(self): return self
        def __exit__(self, *_args): return None
    pipeline = SimpleNamespace(
        uow=UnitOfWork(),
        asset_service=SimpleNamespace(mark_validated=lambda **kwargs: rejected.append(kwargs)),
    )
    executor = SimpleNamespace(
        training_service=SimpleNamespace(run_service=SimpleNamespace(uow=object())),
        motion_pipeline_service=pipeline,
        execute=lambda _payload: (_ for _ in ()).throw(RuntimeError("validation exploded")),
    )
    monkeypatch.setattr(local_worker, "build_executor", lambda: (dispatcher, executor))
    monkeypatch.setattr(local_worker, "OutboxDispatcher", lambda *_args: SimpleNamespace(dispatch=lambda **_kwargs: 0))
    monkeypatch.setattr(local_worker, "LocalRunRecovery", lambda *_args: SimpleNamespace(reconcile=lambda *_args: None))
    monkeypatch.setattr(local_worker.settings, "runtime_root", tmp_path)

    assert local_worker.run(once=True) == 0
    assert failed == ["task-2"]
    assert rejected[0]["rejection_code"] == "VALIDATION_WORKER_FAILED"
    assert "validation exploded" in capsys.readouterr().err


def test_local_worker_marks_motion_failure_in_pipeline(tmp_path, monkeypatch):
    envelope = TaskEnvelope(
        task_id="task-3", queue="motion-gpu", task="allrobotrl.motion.gvhmr",
        payload={"pipeline_id": "pipeline-3"}, idempotency_key="motion-3",
    )
    failures = []
    dispatcher = SimpleNamespace(
        recover_running=lambda: 0, recover_failed=lambda **_kwargs: 0,
        claim=lambda **_kwargs: envelope, complete=lambda *_args: None,
        fail=lambda task_id, **_kwargs: failures.append(task_id),
    )
    pipeline = SimpleNamespace(
        status="QUEUED",
    )
    service = SimpleNamespace(
        store=SimpleNamespace(get=lambda _pipeline_id: pipeline),
        _fail=lambda record, code, message: failures.append((record, code, message)),
    )
    executor = SimpleNamespace(
        training_service=SimpleNamespace(run_service=SimpleNamespace(uow=object())),
        motion_pipeline_service=service,
        execute=lambda _payload: (_ for _ in ()).throw(RuntimeError("runtime unavailable")),
    )
    monkeypatch.setattr(local_worker, "build_executor", lambda: (dispatcher, executor))
    monkeypatch.setattr(local_worker, "OutboxDispatcher", lambda *_args: SimpleNamespace(dispatch=lambda **_kwargs: 0))
    monkeypatch.setattr(local_worker, "LocalRunRecovery", lambda *_args: SimpleNamespace(reconcile=lambda *_args: None))
    monkeypatch.setattr(local_worker.settings, "runtime_root", tmp_path)

    assert local_worker.run(once=True) == 0
    assert failures == [(pipeline, "MOTION_WORKER_FAILED", "runtime unavailable"), "task-3"]
