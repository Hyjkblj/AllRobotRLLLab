from backend.app.application.outbox_dispatcher import OutboxDispatcher, ROUTE_BY_TOPIC
from backend.app.application.run_service import RunService
from backend.app.domain.contracts import Actor
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.infrastructure.queue import InMemoryTaskDispatcher


def test_outbox_dispatcher_publishes_and_marks_events() -> None:
    uow = InMemoryUnitOfWork()
    service = RunService(uow)
    actor = Actor(user_id="alice")
    project = service.create_project(name="P2", actor=actor)
    service.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64)
    dispatcher = InMemoryTaskDispatcher()
    bridge = OutboxDispatcher(uow, dispatcher)
    assert bridge.dispatch() >= 1
    assert not uow.outbox.pending()
    queued = dispatcher.pending()
    assert queued
    assert queued[0].task == "allrobotrl.runs.created"
    assert queued[0].queue == "motion-cpu"


def test_every_outbox_topic_maps_to_a_registered_celery_name() -> None:
    assert ROUTE_BY_TOPIC == {
        "assets.uploading": ("asset-io", "allrobotrl.assets.uploading"),
        "assets.validate": ("asset-io", "allrobotrl.assets.validate"),
        "runs.created": ("motion-cpu", "allrobotrl.runs.created"),
        "runs.retry": ("isaac-gpu", "allrobotrl.runs.retry"),
        "runs.cancelled": ("maintenance", "allrobotrl.runs.cancelled"),
    }
