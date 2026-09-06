"""Production Celery application entry point.

Start with:
``celery -A backend.app.workers.celery_app:celery_app worker -Q isaac-gpu,sim2sim-gpu``
The process requires the same DATABASE_URL/REDIS_URL and object-store
settings as the API so workers operate on durable records and artifacts.
"""

from __future__ import annotations

from backend.app.application.artifact_service import ArtifactService
from backend.app.application.run_service import RunService
from backend.app.application.training_service import TrainingService
from backend.app.application.asset_service import AssetService
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.reward_catalog import RewardConfigVersionStore, RewardRegistry
from backend.app.adapters.motion import MotionSourceRegistry
from backend.app.config.settings import settings
from backend.app.infrastructure.local import build_object_store
from backend.app.infrastructure.postgres_uow import PostgresUnitOfWork
from backend.app.workers.p3_tasks import P3TaskExecutor, register_p3_tasks
from backend.app.application.platform_assembly import build_platform_assembly


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
            "allrobotrl.assets.uploading": {"queue": "asset-io"},
            "allrobotrl.assets.validate": {"queue": "asset-io"},
            "allrobotrl.runs.created": {"queue": "motion-cpu"},
            "allrobotrl.runs.retry": {"queue": "isaac-gpu"},
            "allrobotrl.runs.cancelled": {"queue": "maintenance"},
            "allrobotrl.p3.train": {"queue": "isaac-gpu"},
            "allrobotrl.p3.export": {"queue": "isaac-gpu"},
            "allrobotrl.p3.sim2sim": {"queue": "sim2sim-gpu"},
        },
    )
    uow = PostgresUnitOfWork(settings.database_url)
    run_service = RunService(uow)
    object_store = build_object_store(settings)
    artifact_service = ArtifactService(uow, object_store)
    assembly = build_platform_assembly(settings, workspace=settings.runtime_root / "external")
    deployment_errors = settings.deployment_errors(robot_registry=assembly.robot_registry)
    if deployment_errors:
        raise RuntimeError("Invalid worker deployment configuration: " + "; ".join(deployment_errors))
    robot_registry = assembly.robot_registry
    adapter = assembly.default_adapter
    task_registry = assembly.task_registry
    runtime_adapters = assembly.runtime_adapters
    training_runner = runtime_adapters["providers"].get(settings.p3_backend) if settings.p3_backend in {"native_isaac_lab", "isaac_lab", "unitree_rl_lab"} else None
    if settings.p3_backend == "isaac_lab":
        training_runner = runtime_adapters["providers"].get("native_isaac_lab")
    sim2sim_adapter = runtime_adapters["sim2sim"] if settings.p3_backend in {"native_isaac_lab", "isaac_lab", "unitree_rl_lab", "unitree_mujoco"} else None
    training_providers = runtime_adapters["providers"] if settings.p3_backend != "fake_smoke" else {}
    sim2sim_adapters = runtime_adapters["sim2sim_adapters"] if settings.p3_backend != "fake_smoke" else {}
    reward_config_store = RewardConfigVersionStore(RewardRegistry(), storage_path=settings.runtime_root / "reward_configs.json")
    training_service = TrainingService(run_service=run_service, robot_adapter=adapter, robot_registry=robot_registry, task_registry=task_registry, training_providers=training_providers, sim2sim_adapters=sim2sim_adapters, training_provider_registry=runtime_adapters.get("training_provider_registry"), sim2sim_registry=runtime_adapters.get("sim2sim_registry"), workspace=settings.repository_root / ".runtime" / "p3", artifact_service=artifact_service, object_store=object_store, training_runner=training_runner, sim2sim_adapter=sim2sim_adapter, reward_config_store=reward_config_store)
    motion_pipeline_service = MotionPipelineService(
        uow=uow,
        object_store=object_store,
        robot_adapter=adapter,
        motion_registry=MotionSourceRegistry(),
        motion_editor=assembly.motion_editors[adapter.name],
        asset_service=AssetService(uow, object_store),
        store=MotionPipelineStore(settings.runtime_root / "motion_pipelines"),
        kinematics_compiler=runtime_adapters["compiler"],
        gvhmr_runner=runtime_adapters["gvhmr"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "gmr_gvhmr"} else None,
        gmr_runner=runtime_adapters["gmr"] if settings.p3_backend in {"isaac_lab", "unitree_rl_lab", "gmr_gvhmr"} else None,
        robot_registry=robot_registry,
        motion_editors=assembly.motion_editors,
        kinematics_compilers=runtime_adapters.get("kinematics_compilers", {}),
    )
    register_p3_tasks(celery, P3TaskExecutor(training_service, motion_pipeline_service))
    return celery


celery_app = create_celery_app()

__all__ = ["celery_app", "create_celery_app"]
