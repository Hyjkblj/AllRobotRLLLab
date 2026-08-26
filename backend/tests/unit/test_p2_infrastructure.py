from pathlib import Path

from backend.app.infrastructure.object_store import LocalObjectStore, ObjectStoreError, validate_object_key
from backend.app.infrastructure.queue import CeleryTaskDispatcher, InMemoryTaskDispatcher


def test_object_store_rejects_path_traversal_and_hashes_files(tmp_path: Path) -> None:
    assert validate_object_key("projects/p/assets/a/file.npz") == "projects/p/assets/a/file.npz"
    try:
        validate_object_key("../escape")
    except ObjectStoreError:
        pass
    else:
        raise AssertionError("unsafe object key accepted")
    source = tmp_path / "source.bin"
    source.write_bytes(b"motion")
    store = LocalObjectStore(tmp_path / "objects")
    metadata = store.put_file("projects/p/source.bin", source)
    assert metadata["size_bytes"] == 6
    assert len(metadata["sha256"]) == 64


def test_task_dispatcher_deduplicates_idempotency_keys() -> None:
    dispatcher = InMemoryTaskDispatcher()
    first = dispatcher.enqueue(queue="motion-cpu", task="compile", payload={"run_id": "r"}, idempotency_key="k")
    second = dispatcher.enqueue(queue="motion-cpu", task="compile", payload={"run_id": "r"}, idempotency_key="k")
    assert first == second
    assert len(dispatcher.pending()) == 1


def test_celery_dispatcher_reserves_redis_idempotency_key(monkeypatch) -> None:
    import celery
    import redis

    class FakeRedis:
        values: dict[str, str] = {}

        @classmethod
        def from_url(cls, _url: str, *, decode_responses: bool):
            return cls()

        def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

        def get(self, key: str):
            return self.values.get(key)

        def delete(self, key: str) -> None:
            self.values.pop(key, None)

    class Result:
        def __init__(self, task_id: str) -> None:
            self.id = task_id

    class FakeCelery:
        calls: list[dict] = []

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def send_task(self, task: str, **kwargs):
            self.calls.append({"task": task, **kwargs})
            return Result(kwargs["task_id"])

    monkeypatch.setattr(redis, "Redis", FakeRedis)
    monkeypatch.setattr(celery, "Celery", FakeCelery)
    dispatcher = CeleryTaskDispatcher("redis://test/0")
    first = dispatcher.enqueue(queue="isaac-gpu", task="allrobotrl.p3.train", payload={"run_id": "r"}, idempotency_key="key")
    second = dispatcher.enqueue(queue="isaac-gpu", task="allrobotrl.p3.train", payload={"run_id": "r"}, idempotency_key="key")

    assert first == second
    assert len(FakeCelery.calls) == 1
    assert FakeCelery.calls[0]["task_id"] == first
