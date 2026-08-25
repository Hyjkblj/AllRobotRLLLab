"""Transactional outbox to queue dispatcher bridge."""

from __future__ import annotations

from backend.app.application.run_service import utc_now


QUEUE_BY_TOPIC = {
    "assets.uploading": "asset-io",
    "assets.validate": "asset-io",
    "runs.created": "motion-cpu",
    "runs.retry": "isaac-gpu",
    "runs.cancelled": "maintenance",
}


class OutboxDispatcher:
    def __init__(self, uow, task_dispatcher) -> None:
        self.uow = uow
        self.task_dispatcher = task_dispatcher

    def dispatch(self, *, limit: int = 100) -> int:
        published = 0
        with self.uow:
            events = list(self.uow.outbox.pending(limit=limit))
            for event in events:
                queue = QUEUE_BY_TOPIC.get(event.topic, "maintenance")
                task_id = self.task_dispatcher.enqueue(queue=queue, task=event.topic, payload=event.payload, idempotency_key=event.event_id)
                if task_id:
                    self.uow.outbox.mark_published(event.event_id, utc_now())
                    published += 1
        return published


__all__ = ["OutboxDispatcher", "QUEUE_BY_TOPIC"]

