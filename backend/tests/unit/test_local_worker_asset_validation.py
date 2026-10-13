from types import SimpleNamespace

from backend.app.infrastructure.queue import TaskEnvelope
from backend.app.workers import local_worker


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
