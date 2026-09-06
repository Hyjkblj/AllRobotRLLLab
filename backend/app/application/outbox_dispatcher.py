"""Transactional outbox to queue dispatcher bridge."""

from __future__ import annotations

from backend.app.application.run_service import utc_now


ROUTE_BY_TOPIC = {
    "assets.uploading": ("asset-io", "allrobotrl.assets.uploading"),
    "assets.validate": ("asset-io", "allrobotrl.assets.validate"),
    "runs.created": ("motion-cpu", "allrobotrl.runs.created"),
    "runs.retry": ("isaac-gpu", "allrobotrl.runs.retry"),
    "runs.cancelled": ("maintenance", "allrobotrl.runs.cancelled"),
}

QUEUE_BY_TOPIC = {topic: route[0] for topic, route in ROUTE_BY_TOPIC.items()}


class OutboxDispatcher:
    def __init__(self, uow, task_dispatcher) -> None:
        self.uow = uow
        self.task_dispatcher = task_dispatcher

    def dispatch(self, *, limit: int = 100) -> int:
        published = 0
        with self.uow:
            events = list(self.uow.outbox.pending(limit=limit))
            for event in events:
                route = ROUTE_BY_TOPIC.get(event.topic)
                if route is None:
                    raise ValueError(f"unsupported outbox topic: {event.topic}")
                queue, task = route
                task_id = self.task_dispatcher.enqueue(queue=queue, task=task, payload=event.payload, idempotency_key=event.event_id)
                if task_id:
                    self.uow.outbox.mark_published(event.event_id, utc_now())
                    published += 1
        return published


__all__ = ["OutboxDispatcher", "QUEUE_BY_TOPIC", "ROUTE_BY_TOPIC"]
