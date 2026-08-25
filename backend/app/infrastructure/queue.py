"""Queue port implementations.

The production dispatcher can wrap Celery/Redis.  The local dispatcher keeps
task envelopes inspectable and de-duplicates idempotency keys.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskEnvelope:
    task_id: str
    queue: str
    task: str
    payload: dict
    idempotency_key: str


class InMemoryTaskDispatcher:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: list[TaskEnvelope] = []
        self._by_key: dict[str, TaskEnvelope] = {}

    def enqueue(self, *, queue: str, task: str, payload: dict, idempotency_key: str) -> str:
        with self._lock:
            existing = self._by_key.get(idempotency_key)
            if existing is not None:
                return existing.task_id
            envelope = TaskEnvelope(str(uuid.uuid4()), queue, task, payload, idempotency_key)
            self._tasks.append(envelope)
            self._by_key[idempotency_key] = envelope
            return envelope.task_id

    def pending(self, queue: str | None = None) -> list[TaskEnvelope]:
        with self._lock:
            return [task for task in self._tasks if queue is None or task.queue == queue]


class CeleryTaskDispatcher:
    """Lazy Celery bridge; importing API code does not import Celery."""

    def __init__(self, broker_url: str) -> None:
        try:
            from celery import Celery
        except ImportError as exc:  # pragma: no cover - production worker image only
            raise RuntimeError("Celery is required for the production task dispatcher") from exc
        self._app = Celery("allrobotrl-platform", broker=broker_url)

    def enqueue(self, *, queue: str, task: str, payload: dict, idempotency_key: str) -> str:
        result = self._app.send_task(task, kwargs={"payload": payload, "idempotency_key": idempotency_key}, queue=queue)
        return str(result.id)


__all__ = ["CeleryTaskDispatcher", "InMemoryTaskDispatcher", "TaskEnvelope"]
