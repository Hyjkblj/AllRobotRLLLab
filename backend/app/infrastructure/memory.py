"""Thread-safe local repositories used by development and contract tests.

The repository interfaces mirror the PostgreSQL transaction boundary.  This
module deliberately stores only metadata; large files remain outside this
store and are handled by the object-store port.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime, timezone

from backend.app.domain.contracts import (
    AttemptRecord,
    AssetRecord,
    AssetVersion,
    ArtifactRecord,
    AuditEvent,
    OutboxEvent,
    ProjectMember,
    ProjectRecord,
    P3RunState,
    RunEvent,
    RunRecord,
)


class RepositoryConflict(ValueError):
    """Raised when an immutable or unique record already exists."""


class InMemoryProjectRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._projects: dict[str, ProjectRecord] = {}
        self._members: dict[tuple[str, str], ProjectMember] = {}

    def create(self, project: ProjectRecord, owner: ProjectMember) -> ProjectRecord:
        with self._lock:
            if project.project_id in self._projects:
                raise RepositoryConflict(f"project already exists: {project.project_id}")
            if owner.project_id != project.project_id or owner.user_id != project.owner_id:
                raise ValueError("project owner membership does not match project")
            self._projects[project.project_id] = project
            self._members[(owner.project_id, owner.user_id)] = owner
            return project

    def get(self, project_id: str) -> ProjectRecord | None:
        with self._lock:
            return self._projects.get(project_id)

    def list_for_user(self, user_id: str) -> list[ProjectRecord]:
        with self._lock:
            project_ids = {project_id for project_id, member_user_id in self._members if member_user_id == user_id}
            return [project for project in self._projects.values() if project.project_id in project_ids]

    def member(self, project_id: str, user_id: str) -> ProjectMember | None:
        with self._lock:
            return self._members.get((project_id, user_id))

    def list_members(self, project_id: str) -> list[ProjectMember]:
        with self._lock:
            return [member for (pid, _), member in self._members.items() if pid == project_id]

    def add_member(self, member: ProjectMember) -> ProjectMember:
        with self._lock:
            if member.project_id not in self._projects:
                raise KeyError(f"project not found: {member.project_id}")
            self._members[(member.project_id, member.user_id)] = member
            return member


class InMemoryRunRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._runs: dict[str, RunRecord] = {}
        self._attempts: dict[str, AttemptRecord] = {}
        self._by_key: dict[str, str] = {}

    def create(self, run: RunRecord, attempt: AttemptRecord, *, idempotency_key: str | None = None) -> RunRecord:
        with self._lock:
            if run.run_id in self._runs or attempt.attempt_id in self._attempts:
                raise RepositoryConflict("run or attempt already exists")
            if idempotency_key and idempotency_key in self._by_key:
                raise RepositoryConflict(f"idempotency key already exists: {idempotency_key}")
            if run.current_attempt_id != attempt.attempt_id or attempt.run_id != run.run_id:
                raise ValueError("run current attempt does not match attempt record")
            self._runs[run.run_id] = run
            self._attempts[attempt.attempt_id] = attempt
            if idempotency_key:
                self._by_key[idempotency_key] = run.run_id
            return run

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_for_project(self, project_id: str) -> list[RunRecord]:
        with self._lock:
            return sorted((run for run in self._runs.values() if run.project_id == project_id), key=lambda run: run.updated_at, reverse=True)

    def update(self, run: RunRecord) -> RunRecord:
        with self._lock:
            if run.run_id not in self._runs:
                raise KeyError(f"run not found: {run.run_id}")
            self._runs[run.run_id] = run
            return run

    def attempt(self, attempt_id: str) -> AttemptRecord | None:
        with self._lock:
            return self._attempts.get(attempt_id)

    def list_attempts(self, run_id: str) -> list[AttemptRecord]:
        with self._lock:
            return sorted((attempt for attempt in self._attempts.values() if attempt.run_id == run_id), key=lambda attempt: attempt.number)

    def create_attempt(self, attempt: AttemptRecord) -> AttemptRecord:
        with self._lock:
            if attempt.attempt_id in self._attempts:
                raise RepositoryConflict(f"attempt already exists: {attempt.attempt_id}")
            if any(existing.run_id == attempt.run_id and existing.number == attempt.number for existing in self._attempts.values()):
                raise RepositoryConflict("attempt number already exists for run")
            self._attempts[attempt.attempt_id] = attempt
            return attempt

    def update_attempt(self, attempt: AttemptRecord) -> AttemptRecord:
        with self._lock:
            if attempt.attempt_id not in self._attempts:
                raise KeyError(f"attempt not found: {attempt.attempt_id}")
            self._attempts[attempt.attempt_id] = attempt
            return attempt

    def by_idempotency_key(self, key: str) -> RunRecord | None:
        with self._lock:
            run_id = self._by_key.get(key)
            return self._runs.get(run_id) if run_id else None


class InMemoryEventRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._events: dict[tuple[str, str], list[RunEvent]] = defaultdict(list)

    def append(self, event: RunEvent) -> RunEvent:
        with self._lock:
            key = (event.run_id, event.attempt_id)
            expected = len(self._events[key]) + 1
            if event.seq != expected:
                raise RepositoryConflict(f"event sequence must be contiguous; expected {expected}, got {event.seq}")
            self._events[key].append(event)
            return event

    def list_after(self, run_id: str, attempt_id: str, after_seq: int = 0) -> list[RunEvent]:
        with self._lock:
            return [event for event in self._events[(run_id, attempt_id)] if event.seq > after_seq]


class InMemoryAuditRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> AuditEvent:
        with self._lock:
            self._events.append(event)
            return event

    def list_for_project(self, project_id: str) -> list[AuditEvent]:
        with self._lock:
            return [event for event in self._events if event.project_id == project_id]


class InMemoryOutboxRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._events: dict[str, OutboxEvent] = {}

    def add(self, event: OutboxEvent) -> OutboxEvent:
        with self._lock:
            if event.event_id in self._events:
                raise RepositoryConflict(f"outbox event already exists: {event.event_id}")
            self._events[event.event_id] = event
            return event

    def pending(self, limit: int = 100) -> list[OutboxEvent]:
        with self._lock:
            return [event for event in self._events.values() if event.published_at is None][:limit]

    def mark_published(self, event_id: str, published_at: str) -> OutboxEvent | None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                return None
            updated = event.model_copy(update={"published_at": published_at})
            self._events[event_id] = updated
            return updated


class InMemoryAssetRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._assets: dict[str, AssetRecord] = {}
        self._versions: dict[str, AssetVersion] = {}

    def create(self, asset: AssetRecord, version: AssetVersion) -> tuple[AssetRecord, AssetVersion]:
        with self._lock:
            if asset.asset_id in self._assets or version.asset_version_id in self._versions:
                raise RepositoryConflict("asset or asset version already exists")
            if asset.asset_id != version.asset_id:
                raise ValueError("asset version does not belong to asset")
            self._assets[asset.asset_id] = asset
            self._versions[version.asset_version_id] = version
            return asset, version

    def get(self, asset_id: str) -> AssetRecord | None:
        with self._lock:
            return self._assets.get(asset_id)

    def list_for_project(self, project_id: str) -> list[AssetRecord]:
        with self._lock:
            return sorted((asset for asset in self._assets.values() if asset.project_id == project_id), key=lambda asset: asset.created_at, reverse=True)

    def version(self, asset_version_id: str) -> AssetVersion | None:
        with self._lock:
            return self._versions.get(asset_version_id)

    def update_version(self, version: AssetVersion) -> AssetVersion:
        with self._lock:
            if version.asset_version_id not in self._versions:
                raise KeyError(f"asset version not found: {version.asset_version_id}")
            self._versions[version.asset_version_id] = version
            return version

    def list_versions(self, asset_id: str) -> list[AssetVersion]:
        with self._lock:
            return sorted((version for version in self._versions.values() if version.asset_id == asset_id), key=lambda version: version.version)


class InMemoryArtifactRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._artifacts: dict[str, ArtifactRecord] = {}

    def create(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._lock:
            if artifact.artifact_id in self._artifacts:
                raise RepositoryConflict(f"artifact already exists: {artifact.artifact_id}")
            self._artifacts[artifact.artifact_id] = artifact
            return artifact

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        with self._lock:
            return self._artifacts.get(artifact_id)

    def list_for_run(self, run_id: str) -> list[ArtifactRecord]:
        with self._lock:
            return sorted(
                (artifact for artifact in self._artifacts.values() if artifact.run_id == run_id),
                key=lambda artifact: artifact.created_at,
            )


class InMemoryP3StateRepository:
    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock
        self._states: dict[str, P3RunState] = {}

    def get(self, run_id: str) -> P3RunState | None:
        with self._lock:
            return self._states.get(run_id)

    def upsert(self, state: P3RunState) -> P3RunState:
        with self._lock:
            self._states[state.run_id] = state
            return state


class InMemoryUnitOfWork:
    """Shared-lock unit of work with the same shape as a DB transaction."""

    def __init__(self) -> None:
        lock = threading.RLock()
        self._lock = lock
        self.projects = InMemoryProjectRepository(lock)
        self.runs = InMemoryRunRepository(lock)
        self.events = InMemoryEventRepository(lock)
        self.audits = InMemoryAuditRepository(lock)
        self.outbox = InMemoryOutboxRepository(lock)
        self.assets = InMemoryAssetRepository(lock)
        self.artifacts = InMemoryArtifactRepository(lock)
        self.p3_states = InMemoryP3StateRepository(lock)

    def __enter__(self) -> "InMemoryUnitOfWork":
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


__all__ = ["InMemoryUnitOfWork", "RepositoryConflict"]
