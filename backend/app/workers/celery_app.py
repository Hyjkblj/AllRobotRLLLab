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
from backend.app.application.asset_service import AssetService
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.motion_editor import MotionEditor
from backend.app.adapters.motion import MotionSourceRegistry
from backend.app.config.settings import settings
from backend.app.infrastructure.local import build_object_store
from backend.app.infrastructure.postgres_uow import PostgresUnitOfWork
from backend.app.workers.p3_tasks import P3TaskExecutor, register_p3_tasks
from backend.app.runtime.factory import build_runtime_adapters


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
        task_routes={
            "allrobotrl.motion.process": {"queue": "motion-cpu"},
            "allrobotrl.assets.validate": {"queue": "asset-io"},
            "allrobotrl.p3.train": {"queue": "isaac-gpu"},
            "allrobotrl.p3.export": {"queue": "isaac-gpu"},
            "allrobotrl.p3.sim2sim": {"queue": "sim2sim-gpu"},
        },
    )
    uow = PostgresUnitOfWork(settings.database_url)
    run_service = RunService(uow)
    object_store = build_object_store(settings)
    artifact_service = ArtifactService(uow, object_store)
    adapter = UnitreeG1Adapter(repository_root=settings.repository_root)
    runtime_adapters = build_runtime_adapters(settings, workspace=settings.runtime_root / "external")
    training_runner = runtime_adapters["isaac"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab"} else None
    sim2sim_adapter = runtime_adapters["sim2sim"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "unitree_mujoco"} else None
    training_service = TrainingService(run_service=run_service, robot_adapter=adapter, workspace=settings.repository_root / ".runtime" / "p3", artifact_service=artifact_service, object_store=object_store, training_runner=training_runner, sim2sim_adapter=sim2sim_adapter)
    motion_pipeline_service = MotionPipelineService(
        uow=uow,
        object_store=object_store,
        robot_adapter=adapter,
        motion_registry=MotionSourceRegistry(),
        motion_editor=MotionEditor(adapter.get_spec(), ik_solver=adapter.create_ik_solver()),
        asset_service=AssetService(uow, object_store),
        store=MotionPipelineStore(settings.runtime_root / "motion_pipelines"),
        kinematics_compiler=runtime_adapters["compiler"],
        gvhmr_runner=runtime_adapters["gvhmr"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "gmr_gvhmr"} else None,
        gmr_runner=runtime_adapters["gmr"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "gmr_gvhmr"} else None,
    )
    register_p3_tasks(celery, P3TaskExecutor(training_service, motion_pipeline_service))
    return celery


celery_app = create_celery_app()

__all__ = ["celery_app", "create_celery_app"]
