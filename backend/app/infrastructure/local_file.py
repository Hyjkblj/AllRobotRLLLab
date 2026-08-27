"""Durable Local File Mode repositories.

The local mode deliberately keeps the same repository ports as PostgreSQL,
but stores metadata in a small, inspectable runtime directory.  A process
lock serializes mutations across API, CLI and scheduler processes.  Run data
is kept in independent directories so a damaged cache/index can be rebuilt
without losing manifests, state or event history.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.app.domain.contracts import (
    AssetRecord,
    AssetVersion,
    ArtifactRecord,
    AttemptRecord,
    AuditEvent,
    OutboxEvent,
    P3RunState,
    ProjectMember,
    ProjectRecord,
    RunEvent,
    RunRecord,
)
from backend.app.infrastructure.memory import InMemoryUnitOfWork


class LocalStateError(RuntimeError):
    """Raised when a durable local state file cannot be decoded."""


class FileLock:
    """Cross-process advisory lock with a Windows fallback.

    WSL/Linux uses ``fcntl.flock``.  The fallback is intentionally small and
    only used for native Windows development where the API is single-user.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None
        self._thread_lock = threading.RLock()
        self._depth = 0

    def acquire(self) -> None:
        with self._thread_lock:
            if self._depth:
                self._depth += 1
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    handle.write(b"0")
                    handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                handle.close()
                raise
            self._handle = handle
            self._depth = 1

    def release(self) -> None:
        with self._thread_lock:
            if not self._depth:
                return
            self._depth -= 1
            if self._depth:
                return
            handle = self._handle
            self._handle = None
            if handle is None:
                return
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    _atomic_write_bytes(path, payload + b"\n")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalStateError(f"invalid local state file: {path}") from exc


def _model(value: Any, model_type, path: Path):
    try:
        return model_type.model_validate(value)
    except Exception as exc:
        raise LocalStateError(f"invalid {model_type.__name__} in {path}") from exc


class LocalFileUnitOfWork(InMemoryUnitOfWork):
    """Persistent UnitOfWork compatible with the in-memory and PostgreSQL ports."""

    schema_version = "local_file_state.v1"

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.runs_root = self.root / "runs"
        self.scheduler_root = self.root / "scheduler"
        self.lock = FileLock(self.scheduler_root / "state.lock")
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.scheduler_root.mkdir(parents=True, exist_ok=True)
        super().__init__()
        self._depth = 0
        self._load()

    def __enter__(self) -> "LocalFileUnitOfWork":
        if self._depth == 0:
            self.lock.acquire()
            self._lock.acquire()
            try:
                self._load()
            except Exception:
                self._lock.release()
                self.lock.release()
                raise
        else:
            self._lock.acquire()
        self._depth += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._depth -= 1
        try:
            if self._depth == 0:
                if exc_type is None:
                    self._persist()
                else:
                    # Roll back the in-process view to the last durable state.
                    self._load()
        finally:
            self._lock.release()
            if self._depth == 0:
                self.lock.release()

    @contextmanager
    def run_lock(self, run_id: str) -> Iterator[FileLock]:
        """Lock one Run directory for scheduler/worker side effects."""

        path = self.runs_root / run_id / "lock"
        run_lock = FileLock(path)
        with run_lock:
            yield run_lock

    def rebuild_index(self) -> dict[str, Any]:
        """Rebuild the derived index from durable records."""

        with self._lock:
            self._load()
            index = self._index_payload()
            _atomic_write_json(self.root / "index.json", index)
            return index

    def _load(self) -> None:
        self._clear_repositories()
        self._load_projects()
        self._load_assets()
        self._load_runs()
        self._load_global("audits.json", self.audits._events, AuditEvent)
        outbox = _read_json(self.root / "outbox.json", [])
        for item in outbox:
            event = _model(item, OutboxEvent, self.root / "outbox.json")
            self.outbox._events[event.event_id] = event
        artifacts = _read_json(self.root / "artifacts.json", [])
        for item in artifacts:
            artifact = _model(item, ArtifactRecord, self.root / "artifacts.json")
            self.artifacts._artifacts[artifact.artifact_id] = artifact
        self._load_index_or_rebuild()

    def _clear_repositories(self) -> None:
        self.projects._projects.clear()
        self.projects._members.clear()
        self.runs._runs.clear()
        self.runs._attempts.clear()
        self.runs._by_key.clear()
        self.events._events.clear()
        self.audits._events.clear()
        self.outbox._events.clear()
        self.assets._assets.clear()
        self.assets._versions.clear()
        self.artifacts._artifacts.clear()
        self.p3_states._states.clear()

    def _load_projects(self) -> None:
        path = self.root / "projects.json"
        payload = _read_json(path, {"projects": [], "members": []})
        for item in payload.get("projects", []):
            project = _model(item, ProjectRecord, path)
            self.projects._projects[project.project_id] = project
        for item in payload.get("members", []):
            member = _model(item, ProjectMember, path)
            self.projects._members[(member.project_id, member.user_id)] = member

    def _load_assets(self) -> None:
        path = self.root / "assets.json"
        payload = _read_json(path, {"assets": [], "versions": []})
        for item in payload.get("assets", []):
            asset = _model(item, AssetRecord, path)
            self.assets._assets[asset.asset_id] = asset
        for item in payload.get("versions", []):
            version = _model(item, AssetVersion, path)
            self.assets._versions[version.asset_version_id] = version

    def _load_runs(self) -> None:
        if not self.runs_root.exists():
            return
        for run_dir in sorted(path for path in self.runs_root.iterdir() if path.is_dir()):
            state_path = run_dir / "state.json"
            if not state_path.exists():
                continue
            payload = _read_json(state_path, {})
            run = _model(payload.get("run"), RunRecord, state_path)
            self.runs._runs[run.run_id] = run
            for item in payload.get("attempts", []):
                attempt = _model(item, AttemptRecord, state_path)
                self.runs._attempts[attempt.attempt_id] = attempt
            for key, run_id in payload.get("idempotency_keys", {}).items():
                self.runs._by_key[str(key)] = str(run_id)
            p3_payload = payload.get("p3_state")
            if p3_payload:
                state = _model(p3_payload, P3RunState, state_path)
                self.p3_states._states[state.run_id] = state
            events_path = run_dir / "events.jsonl"
            if events_path.exists():
                lines = events_path.read_text(encoding="utf-8").splitlines()
                for line_number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        # A torn final append is recoverable after a crash.
                        if line_number == len(lines):
                            break
                        raise LocalStateError(f"invalid event in {events_path}:{line_number}") from exc
                    event = _model(value, RunEvent, events_path)
                    self.events._events[(event.run_id, event.attempt_id)].append(event)

    def _load_global(self, filename: str, target: list, model_type) -> None:
        path = self.root / filename
        for item in _read_json(path, []):
            target.append(_model(item, model_type, path))

    def _load_index_or_rebuild(self) -> None:
        index_path = self.root / "index.json"
        if not index_path.exists():
            _atomic_write_json(index_path, self._index_payload())
            return
        try:
            value = json.loads(index_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != self.schema_version:
                raise ValueError("schema mismatch")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError):
            _atomic_write_json(index_path, self._index_payload())

    def _persist(self) -> None:
        projects = list(self.projects._projects.values())
        members = list(self.projects._members.values())
        _atomic_write_json(self.root / "projects.json", {"projects": [item.model_dump(mode="json") for item in projects], "members": [item.model_dump(mode="json") for item in members]})
        _atomic_write_json(self.root / "assets.json", {"assets": [item.model_dump(mode="json") for item in self.assets._assets.values()], "versions": [item.model_dump(mode="json") for item in self.assets._versions.values()]})
        _atomic_write_json(self.root / "audits.json", [item.model_dump(mode="json") for item in self.audits._events])
        _atomic_write_json(self.root / "outbox.json", [item.model_dump(mode="json") for item in self.outbox._events.values()])
        _atomic_write_json(self.root / "artifacts.json", [item.model_dump(mode="json") for item in self.artifacts._artifacts.values()])
        for run_id, run in self.runs._runs.items():
            run_dir = self.runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "lock").touch(exist_ok=True)
            attempts = self.runs.list_attempts(run_id)
            p3_state = self.p3_states._states.get(run_id)
            payload = {
                "schema_version": self.schema_version,
                "run": run.model_dump(mode="json"),
                "attempts": [item.model_dump(mode="json") for item in attempts],
                "idempotency_keys": {key: value for key, value in self.runs._by_key.items() if value == run_id},
                "p3_state": p3_state.model_dump(mode="json") if p3_state else None,
            }
            _atomic_write_json(run_dir / "state.json", payload)
            _atomic_write_json(run_dir / "manifest.json", run.manifest.model_dump(mode="json"))
            events = []
            for key_events in self.events._events.values():
                events.extend(event for event in key_events if event.run_id == run_id)
            _append_event_log(run_dir / "events.jsonl", events)
            metrics = [event for event in events if event.event_type == "metric"]
            _append_event_log(run_dir / "metrics.jsonl", metrics)
        _atomic_write_json(self.root / "index.json", self._index_payload())

    def _index_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "projects": [item.model_dump(mode="json") for item in self.projects._projects.values()],
            "runs": [
                {"run_id": run.run_id, "project_id": run.project_id, "status": run.status.value, "updated_at": run.updated_at}
                for run in self.runs._runs.values()
            ],
            "artifact_count": len(self.artifacts._artifacts),
        }


def _append_event_log(path: Path, events: list[RunEvent]) -> None:
    """Append new records and rewrite only if the existing log diverged."""

    encoded = [json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for event in events]
    existing = path.read_bytes() if path.exists() else b""
    existing_lines = existing.splitlines(keepends=True)
    prefix_matches = True
    if len(existing_lines) > len(encoded):
        prefix_matches = False
    else:
        prefix_matches = all(existing_lines[index].rstrip(b"\r\n") == encoded[index].rstrip(b"\r\n") for index in range(len(existing_lines)))
    if prefix_matches:
        tail = b"".join(encoded[len(existing_lines):])
        if tail:
            with path.open("ab") as stream:
                stream.write(tail)
                stream.flush()
                os.fsync(stream.fileno())
        return
    _atomic_write_bytes(path, b"".join(encoded))


__all__ = ["FileLock", "LocalFileUnitOfWork", "LocalStateError"]
