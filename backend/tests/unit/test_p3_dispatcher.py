from pathlib import Path

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.p3_dispatcher import P3DispatchService
from backend.app.application.run_service import RunService
from backend.app.application.training_service import TrainingService
from backend.app.domain.contracts import Actor, TrainingConfig
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.infrastructure.queue import InMemoryTaskDispatcher
from backend.app.workers.p3_tasks import P3TaskExecutor


def test_p3_async_submission_is_idempotent_and_worker_can_execute(tmp_path: Path) -> None:
    uow = InMemoryUnitOfWork()
    run_service = RunService(uow)
    actor = Actor(user_id="alice")
    project = run_service.create_project(name="P3 async", actor=actor)
    run, _, _ = run_service.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64)
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    training = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=tmp_path)
    queue = InMemoryTaskDispatcher()
    dispatch = P3DispatchService(run_service=run_service, training_service=training, task_dispatcher=queue)
    config = TrainingConfig(motion_asset_version_id="motion-1")

    first = dispatch.submit_train(run_id=run.run_id, config=config, actor=actor, worker_id="worker-1")
    replay = dispatch.submit_train(run_id=run.run_id, config=config, actor=actor, worker_id="worker-1")
    assert first.task_id == replay.task_id
    assert first.idempotency_key == replay.idempotency_key
    assert len(queue.pending("isaac-gpu")) == 1

    envelope = queue.pending("isaac-gpu")[0]
    result = P3TaskExecutor(training).execute({**envelope.payload, "operation": "train"})
    assert result["status"] == "SUCCEEDED"
    final, _ = run_service.get_run(run_id=run.run_id, actor=actor)
    assert final.status == "TRAINING_SUCCEEDED"
