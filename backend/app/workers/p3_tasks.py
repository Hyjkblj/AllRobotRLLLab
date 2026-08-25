"""Worker-facing entry points for P3 queue envelopes.

The executor is deliberately dependency-injected. A production worker can
construct it with PostgreSQL-backed repositories and the real Isaac/MuJoCo
adapters, while unit tests use the in-memory application services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.application.training_service import TrainingService
from backend.app.domain.contracts import Sim2SimThresholds, TrainingConfig


@dataclass
class P3TaskExecutor:
    training_service: TrainingService

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        operation = str(payload.get("operation", "")).strip().lower()
        run_id = str(payload["run_id"])
        if operation == "train":
            config = TrainingConfig.model_validate(payload["config"])
            result = self.training_service.train_smoke(run_id=run_id, config=config, worker_id=str(payload.get("worker_id", "celery-worker")))
            return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "checkpoint_id": result.checkpoint.checkpoint_id}
        if operation == "export":
            bundle = self.training_service.export(run_id=run_id)
            return {"operation": operation, "run_id": run_id, "status": "SUCCEEDED", "bundle_id": bundle.bundle_id}
        if operation == "sim2sim":
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


__all__ = ["P3TaskExecutor", "register_p3_tasks"]
