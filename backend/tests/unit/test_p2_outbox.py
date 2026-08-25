from backend.app.application.outbox_dispatcher import OutboxDispatcher
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
    assert dispatcher.pending()
