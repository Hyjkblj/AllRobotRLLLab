from pathlib import Path

from backend.app.infrastructure.object_store import LocalObjectStore, ObjectStoreError, validate_object_key
from backend.app.infrastructure.queue import InMemoryTaskDispatcher


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

