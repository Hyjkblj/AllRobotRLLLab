"""Production Celery application entry point.

Start with:
``celery -A backend.app.workers.celery_app:celery_app worker -Q isaac-gpu,sim2sim-gpu``
The process requires the same DATABASE_URL/REDIS_URL and object-store
settings as the API so workers operate on durable records and artifacts.
"""

from __future__ import annotations

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.artifact_service import ArtifactService
from backend.app.application.run_service import RunService
from backend.app.application.training_service import TrainingService
from backend.app.config.settings import settings
from backend.app.infrastructure.local import build_object_store
from backend.app.infrastructure.postgres_uow import PostgresUnitOfWork
from backend.app.workers.p3_tasks import P3TaskExecutor, register_p3_tasks


def create_celery_app():
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for a production Celery worker")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required for a production Celery worker")
    from celery import Celery

    celery = Celery("allrobotrl-platform", broker=settings.redis_url, backend=settings.redis_url)
    celery.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        worker_prefetch_multiplier=1,
        result_expires=86400,
    )
    uow = PostgresUnitOfWork(settings.database_url)
    run_service = RunService(uow)
    object_store = build_object_store(settings)
    artifact_service = ArtifactService(uow, object_store)
    adapter = UnitreeG1Adapter(repository_root=settings.repository_root)
    training_service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=settings.repository_root / ".runtime" / "p3", artifact_service=artifact_service, object_store=object_store)
    register_p3_tasks(celery, P3TaskExecutor(training_service))
    return celery


celery_app = create_celery_app()

__all__ = ["celery_app", "create_celery_app"]
