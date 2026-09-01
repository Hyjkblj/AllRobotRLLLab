"""Worker-facing entry points for P3 queue envelopes.

The executor is deliberately dependency-injected. A production worker can
construct it with PostgreSQL-backed repositories and the real Isaac/MuJoCo
adapters, while unit tests use the in-memory application services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.application.training_service import TrainingService
from backend.app.application.motion_pipeline_service import MotionPipelineService
from backend.app.config.settings import settings
from backend.app.domain.contracts import Sim2SimThresholds, TrainingConfig
from backend.app.domain.state_machine import RunStatus


@dataclass
class P3TaskExecutor:
    training_service: TrainingService
    motion_pipeline_service: MotionPipelineService | None = None

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        operation = str(payload.get("operation", "")).strip().lower()
        run_id = str(payload.get("run_id", ""))
        if operation == "asset_validate":
            if self.motion_pipeline_service is None:
                raise RuntimeError("asset validation service is not configured")
            version = self.motion_pipeline_service.validate_asset_version(str(payload["asset_version_id"]))
            return {"operation": operation, "asset_version_id": version.asset_version_id, "status": version.status.value, "rejection_code": version.rejection_code}
        if operation == "motion_process":
            if self.motion_pipeline_service is None:
                raise RuntimeError("motion pipeline service is not configured")
            record = self.motion_pipeline_service.process(str(payload["pipeline_id"]))
            return {"operation": operation, "pipeline_id": record.pipeline_id, "status": record.status, "output_asset_version_id": record.output_asset_version_id, "error_code": record.error_code}
        if operation == "retry":
            config_payload = payload.get("config")
            if config_payload:
                config = TrainingConfig.model_validate(config_payload)
                result = self.training_service.train(run_id=run_id, config=config, worker_id=str(payload.get("worker_id", "local-worker")))
                return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "checkpoint_id": result.checkpoint.checkpoint_id}
            return {"operation": operation, "run_id": run_id, "status": "ACKNOWLEDGED", "message": "no prior training config was available"}
        if operation in {"run_validate", "asset_uploading", "cancelled"}:
            return {"operation": operation, "run_id": str(payload.get("run_id", "")), "status": "ACKNOWLEDGED"}
        if not run_id:
            raise ValueError("run_id is required")
        if settings.is_deployed and (self.training_service is None or getattr(self.training_service, "training_runner", None) is None):
            raise RuntimeError(f"real P3 backend '{settings.p3_backend}' is not registered in this platform image; refusing to run CPU smoke in a deployed environment")
        with self.training_service.run_service.uow:
            run = self.training_service.run_service.uow.runs.get(run_id)
            state = self.training_service.run_service.uow.p3_states.get(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        if run.status == RunStatus.CANCELLED:
            # A queued message can outlive a user cancellation.  Acknowledge
            # it without starting a simulator or creating new artifacts.
            return {"operation": operation, "run_id": run_id, "status": "CANCELLED", "cancelled": True}
        if operation == "train":
            if state is not None and state.checkpoint is not None and run.status in {RunStatus.TRAINING_SUCCEEDED, RunStatus.EXPORTING, RunStatus.EXPORTED, RunStatus.SIM2SIM_QUEUED, RunStatus.SIM2SIM_RUNNING, RunStatus.SIM2SIM_PASSED, RunStatus.READY_TO_DOWNLOAD}:
                return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "checkpoint_id": state.checkpoint.checkpoint_id, "replayed": True}
            config = TrainingConfig.model_validate(payload["config"])
            result = self.training_service.train(run_id=run_id, config=config, worker_id=str(payload.get("worker_id", "celery-worker")))
            return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "checkpoint_id": result.checkpoint.checkpoint_id}
        if operation == "export":
            if state is not None and state.bundle is not None and run.status in {RunStatus.EXPORTED, RunStatus.SIM2SIM_QUEUED, RunStatus.SIM2SIM_RUNNING, RunStatus.SIM2SIM_PASSED, RunStatus.READY_TO_DOWNLOAD}:
                return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "bundle_id": state.bundle.bundle_id, "replayed": True}
            bundle = self.training_service.export(run_id=run_id)
            return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "bundle_id": bundle.bundle_id}
        if operation == "sim2sim":
            if state is not None and state.sim2sim_report is not None and run.status in {RunStatus.SIM2SIM_PASSED, RunStatus.READY_TO_DOWNLOAD, RunStatus.FAILED}:
                return {"operation": operation, "run_id": run_id, "status": state.sim2sim_report.status, "report_sha256": state.sim2sim_report.report_sha256, "replayed": True}
            thresholds = payload.get("thresholds")
            report = self.training_service.sim2sim(run_id=run_id, seeds=tuple(int(seed) for seed in payload.get("seeds", ())), thresholds=Sim2SimThresholds.model_validate(thresholds) if thresholds else None)
            return {"operation": operation, "run_id": run_id, "status": report.status, "report_sha256": report.report_sha256}
        raise ValueError(f"unsupported P3 task operation: {operation}")


def register_p3_tasks(celery_app, executor: P3TaskExecutor) -> None:
    """Register stable Celery names on an application created by deployment."""

    @celery_app.task(name="allrobotrl.p3.train")
    def train(payload: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        return executor.execute({**payload, "operation": "train"})

    @celery_app.task(name="allrobotrl.p3.export")
    def export(payload: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        return executor.execute({**payload, "operation": "export"})

    @celery_app.task(name="allrobotrl.p3.sim2sim")
    def sim2sim(payload: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        return executor.execute({**payload, "operation": "sim2sim"})

    @celery_app.task(name="allrobotrl.motion.process")
    def motion_process(payload: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        return executor.execute({**payload, "operation": "motion_process"})

    @celery_app.task(name="allrobotrl.assets.validate")
    def asset_validate(payload: dict[str, Any], idempotency_key: str | None = None) -> dict[str, Any]:
        return executor.execute({**payload, "operation": "asset_validate"})


__all__ = ["P3TaskExecutor", "register_p3_tasks"]
