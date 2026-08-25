"""Asynchronous submission boundary for the P3 long-running stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from backend.app.application.run_service import RunService, RunServiceError
from backend.app.application.training_service import TrainingService, TrainingServiceError
from backend.app.domain.contracts import Actor, Sim2SimThresholds, TaskSubmission, TrainingConfig
from backend.app.domain.state_machine import RunStatus


class P3DispatchError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class P3DispatchService:
    """Translate API operations into idempotent queue envelopes.

    The dispatcher owns no training logic. Workers call ``TrainingService``
    after consuming these envelopes, which keeps HTTP requests short and
    leaves the same application service usable by the local smoke backend.
    """

    run_service: RunService
    training_service: TrainingService
    task_dispatcher: object

    def submit_train(self, *, run_id: str, config: TrainingConfig, actor: Actor, worker_id: str) -> TaskSubmission:
        try:
            run, attempt = self.training_service.prepare_training(run_id=run_id, config=config, actor=actor)
        except (RunServiceError, TrainingServiceError) as exc:
            raise P3DispatchError(getattr(exc, "code", "TRAIN_SUBMISSION_INVALID"), str(exc), status_code=getattr(exc, "status_code", 400)) from exc
        payload = {"run_id": run_id, "attempt_id": attempt.attempt_id, "config": config.model_dump(mode="json"), "worker_id": worker_id}
        return self._enqueue(operation="train", queue="isaac-gpu", task="allrobotrl.p3.train", run_id=run_id, attempt_id=attempt.attempt_id, payload=payload)

    def submit_export(self, *, run_id: str, actor: Actor, worker_id: str) -> TaskSubmission:
        run, attempts = self._get_run(run_id=run_id, actor=actor)
        if run.status != RunStatus.TRAINING_SUCCEEDED:
            raise P3DispatchError("EXPORT_STATUS_INVALID", f"run must be TRAINING_SUCCEEDED, got {run.status}", status_code=409)
        attempt = next(attempt for attempt in attempts if attempt.attempt_id == run.current_attempt_id)
        payload = {"run_id": run_id, "attempt_id": attempt.attempt_id, "worker_id": worker_id}
        return self._enqueue(operation="export", queue="isaac-gpu", task="allrobotrl.p3.export", run_id=run_id, attempt_id=attempt.attempt_id, payload=payload)

    def submit_sim2sim(self, *, run_id: str, seeds: tuple[int, int, int], thresholds: Sim2SimThresholds | None, actor: Actor, worker_id: str) -> TaskSubmission:
        run, attempts = self._get_run(run_id=run_id, actor=actor)
        if run.status != RunStatus.EXPORTED:
            raise P3DispatchError("SIM2SIM_STATUS_INVALID", f"run must be EXPORTED, got {run.status}", status_code=409)
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise P3DispatchError("SIM2SIM_SEED_INVALID", "exactly three distinct seeds are required")
        attempt = next(attempt for attempt in attempts if attempt.attempt_id == run.current_attempt_id)
        payload = {"run_id": run_id, "attempt_id": attempt.attempt_id, "seeds": list(seeds), "thresholds": thresholds.model_dump(mode="json") if thresholds else None, "worker_id": worker_id}
        return self._enqueue(operation="sim2sim", queue="sim2sim-gpu", task="allrobotrl.p3.sim2sim", run_id=run_id, attempt_id=attempt.attempt_id, payload=payload)

    def _get_run(self, *, run_id: str, actor: Actor):
        try:
            return self.run_service.get_run(run_id=run_id, actor=actor)
        except RunServiceError as exc:
            raise P3DispatchError(exc.code, exc.message, status_code=exc.status_code) from exc

    def _enqueue(self, *, operation: str, queue: str, task: str, run_id: str, attempt_id: str, payload: dict) -> TaskSubmission:
        canonical = json.dumps({"operation": operation, "run_id": run_id, "attempt_id": attempt_id, "payload": payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        idempotency_key = hashlib.sha256(canonical).hexdigest()
        try:
            task_id = self.task_dispatcher.enqueue(queue=queue, task=task, payload=payload, idempotency_key=idempotency_key)
        except Exception as exc:
            raise P3DispatchError("TASK_QUEUE_UNAVAILABLE", f"unable to enqueue {operation}: {exc}", status_code=503) from exc
        return TaskSubmission(task_id=task_id, run_id=run_id, attempt_id=attempt_id, operation=operation, queue=queue, idempotency_key=idempotency_key)


__all__ = ["P3DispatchError", "P3DispatchService"]
