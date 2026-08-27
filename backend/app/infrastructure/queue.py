"""Queue port implementations.

The production dispatcher can wrap Celery/Redis.  The local dispatcher keeps
task envelopes inspectable and de-duplicates idempotency keys.
"""

from __future__ import annotations

import threading
import uuid
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from backend.app.infrastructure.local_file import FileLock


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


class LocalFileTaskDispatcher:
    """Durable local task queue used by Local File Mode.

    Queue records are rewritten atomically under a process lock.  The queue is
    intentionally transport-neutral: a local scheduler can claim records,
    while Compose mode continues to use the Celery implementation below.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "tasks.json"
        self.lock = FileLock(self.root / "queue.lock")
        self._thread_lock = threading.RLock()

    def enqueue(self, *, queue: str, task: str, payload: dict, idempotency_key: str) -> str:
        with self._locked():
            records = self._read()
            for record in records:
                if record["idempotency_key"] == idempotency_key:
                    return str(record["task_id"])
            task_id = str(uuid.uuid4())
            records.append({"task_id": task_id, "queue": queue, "task": task, "payload": payload, "idempotency_key": idempotency_key, "status": "QUEUED", "attempts": 0})
            self._write(records)
            return task_id

    def pending(self, queue: str | None = None) -> list[TaskEnvelope]:
        with self._locked():
            return [self._envelope(record) for record in self._read() if record["status"] == "QUEUED" and (queue is None or record["queue"] == queue)]

    def claim(self, *, worker_id: str, queues: tuple[str, ...] | None = None) -> TaskEnvelope | None:
        if not worker_id.strip():
            raise ValueError("worker_id is required")
        with self._locked():
            records = self._read()
            for record in records:
                if record["status"] != "QUEUED" or (queues and record["queue"] not in queues):
                    continue
                record["status"] = "RUNNING"
                record["worker_id"] = worker_id
                record["attempts"] = int(record.get("attempts", 0)) + 1
                self._write(records)
                return self._envelope(record)
        return None

    def complete(self, task_id: str) -> bool:
        return self._set_status(task_id, "SUCCEEDED")

    def fail(self, task_id: str, *, retry: bool = True) -> bool:
        return self._set_status(task_id, "QUEUED" if retry else "FAILED")

    def recover_running(self) -> int:
        """Requeue tasks left RUNNING by a crashed local scheduler."""

        with self._locked():
            records = self._read()
            changed = 0
            for record in records:
                if record["status"] == "RUNNING":
                    record["status"] = "QUEUED"
                    record.pop("worker_id", None)
                    changed += 1
            if changed:
                self._write(records)
            return changed

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            with self.lock:
                yield

    def _set_status(self, task_id: str, status: str) -> bool:
        with self._locked():
            records = self._read()
            for record in records:
                if record["task_id"] == task_id:
                    record["status"] = status
                    self._write(records)
                    return True
        return False

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid local task queue: {self.path}") from exc
        if not isinstance(value, list):
            raise RuntimeError(f"invalid local task queue: {self.path}")
        return value

    def _write(self, records: list[dict]) -> None:
        payload = json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".tasks.", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _envelope(record: dict) -> TaskEnvelope:
        return TaskEnvelope(task_id=str(record["task_id"]), queue=str(record["queue"]), task=str(record["task"]), payload=dict(record.get("payload", {})), idempotency_key=str(record["idempotency_key"]))


class CeleryTaskDispatcher:
    """Lazy Celery bridge; importing API code does not import Celery."""

    def __init__(self, broker_url: str) -> None:
        try:
            from celery import Celery
            from redis import Redis
        except ImportError as exc:  # pragma: no cover - production worker image only
            raise RuntimeError("Celery and redis are required for the production task dispatcher") from exc
        self._app = Celery("allrobotrl-platform", broker=broker_url)
        self._redis = Redis.from_url(broker_url, decode_responses=True)
        self._idempotency_ttl = 7 * 24 * 60 * 60

    def enqueue(self, *, queue: str, task: str, payload: dict, idempotency_key: str) -> str:
        redis_key = f"allrobotrl:task-idempotency:{idempotency_key}"
        task_id = str(uuid.uuid4())
        if not self._redis.set(redis_key, task_id, nx=True, ex=self._idempotency_ttl):
            existing = self._redis.get(redis_key)
            if existing:
                return str(existing)
            # The key can expire between SETNX and GET; retry once with a new
            # token rather than enqueueing an untracked duplicate.
            if not self._redis.set(redis_key, task_id, nx=True, ex=self._idempotency_ttl):
                existing = self._redis.get(redis_key)
                if existing:
                    return str(existing)
                raise RuntimeError("task idempotency key could not be reserved")
        try:
            result = self._app.send_task(task, kwargs={"payload": payload, "idempotency_key": idempotency_key}, queue=queue, task_id=task_id)
        except Exception:
            # A failed enqueue must be retryable. A concurrent caller that
            # already replaced this token is left untouched.
            if self._redis.get(redis_key) == task_id:
                self._redis.delete(redis_key)
            raise
        return str(result.id)


__all__ = ["CeleryTaskDispatcher", "InMemoryTaskDispatcher", "LocalFileTaskDispatcher", "TaskEnvelope"]
