from backend.app.application.run_service import RunService, RunServiceError
from backend.app.domain.contracts import Actor, ProjectRole, RunStatus
from backend.app.infrastructure.memory import InMemoryUnitOfWork


def _service() -> tuple[RunService, Actor]:
    return RunService(InMemoryUnitOfWork()), Actor(user_id="alice", email="alice@example.com")


def _run_values(project_id: str) -> dict:
    return {"project_id": project_id, "robot": {"robot_id": "unitree_g1_29dof", "adapter_sha": "adapter"}, "motion": {"input_mode": "direct", "train_motion_sha256": "a" * 64}, "reward_config_sha256": "b" * 64, "training_config_sha256": "c" * 64}


def test_project_permissions_and_run_idempotency() -> None:
    service, alice = _service()
    project = service.create_project(name="P2", actor=alice)
    service.add_member(project_id=project.project_id, actor=alice, user_id="bob", role=ProjectRole.VIEWER)
    first, attempt, replayed = service.create_run(actor=alice, idempotency_key="request-1", **_run_values(project.project_id))
    same, same_attempt, replayed = service.create_run(actor=alice, idempotency_key="request-1", **_run_values(project.project_id))
    assert first.run_id == same.run_id
    assert attempt.attempt_id == same_attempt.attempt_id
    assert replayed
    fetched, _ = service.get_run(run_id=first.run_id, actor=Actor(user_id="bob"))
    assert fetched.run_id == first.run_id
    try:
        service.cancel_run(run_id=first.run_id, actor=Actor(user_id="bob"))
    except RunServiceError as error:
        assert error.code == "PROJECT_ACCESS_DENIED"
    else:
        raise AssertionError("viewer was able to cancel a run")


def test_run_events_cancel_and_retry_preserve_attempt_history() -> None:
    service, alice = _service()
    project = service.create_project(name="P2", actor=alice)
    run, attempt, _ = service.create_run(actor=alice, **_run_values(project.project_id))
    events = service.list_events(run_id=run.run_id, actor=alice)
    assert events[0].seq == 1
    cancelled, cancelled_attempt = service.cancel_run(run_id=run.run_id, actor=alice)
    assert cancelled.status == RunStatus.CANCELLED
    assert cancelled_attempt.status == RunStatus.CANCELLED

    # Create a second run and drive it to FAILED through the explicit state machine.
    retryable, _, _ = service.create_run(actor=alice, **_run_values(project.project_id))
    service.transition_run(run_id=retryable.run_id, target=RunStatus.VALIDATING)
    service.transition_run(run_id=retryable.run_id, target=RunStatus.FAILED)
    retried, new_attempt = service.retry_run(run_id=retryable.run_id, actor=alice)
    assert retried.status == RunStatus.TRAINING_PREPARING
    assert new_attempt.number == 2
    assert new_attempt.attempt_id != retryable.current_attempt_id


def test_worker_heartbeat_updates_current_attempt() -> None:
    service, alice = _service()
    project = service.create_project(name="P2", actor=alice)
    run, _, _ = service.create_run(actor=alice, **_run_values(project.project_id))
    heartbeat = service.heartbeat(run_id=run.run_id, worker_id="worker-1", gpu_uuid="GPU-0")
    assert heartbeat.worker_id == "worker-1"
    assert heartbeat.gpu_uuid == "GPU-0"
    assert heartbeat.last_heartbeat_at
