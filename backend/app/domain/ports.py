"""Ports implemented by adapters and infrastructure at the domain boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from pathlib import Path

from .contracts import (
    RetargetMotion,
    RobotSpec,
    RunManifest,
    SourceMotionDescriptor,
    TrainMotionNPZ,
    TrainingConfig,
    ValidationResult,
)
from .contracts import (
    Actor,
    AssetRecord,
    AssetVersion,
    ArtifactRecord,
    AttemptRecord,
    AuditEvent,
    OutboxEvent,
    ProjectMember,
    ProjectRecord,
    ProjectRole,
    RunEvent,
    RunRecord,
)


class MotionSourceAdapter(Protocol):
    name: str
    version: str

    def detect(self, path: Path) -> SourceMotionDescriptor: ...

    def validate(self, descriptor: SourceMotionDescriptor) -> ValidationResult: ...


class RobotAdapter(Protocol):
    def get_spec(self) -> RobotSpec: ...

    def self_check(self) -> ValidationResult: ...

    def validate_motion(self, motion: RetargetMotion) -> ValidationResult: ...

    def compile_motion(self, motion: RetargetMotion, config: TrainingConfig, output_dir: Path) -> TrainMotionNPZ: ...

    def validate_training_manifest(self, manifest: RunManifest) -> ValidationResult: ...


class TrainingBackendAdapter(Protocol):
    def validate_config(self, manifest: RunManifest) -> ValidationResult: ...


class Sim2SimAdapter(Protocol):
    def validate_bundle(self, manifest: RunManifest) -> ValidationResult: ...


class ProjectRepository(Protocol):
    def create(self, project: ProjectRecord, owner: ProjectMember) -> ProjectRecord: ...

    def get(self, project_id: str) -> ProjectRecord | None: ...

    def list_for_user(self, user_id: str) -> list[ProjectRecord]: ...

    def member(self, project_id: str, user_id: str) -> ProjectMember | None: ...

    def list_members(self, project_id: str) -> list[ProjectMember]: ...

    def add_member(self, member: ProjectMember) -> ProjectMember: ...


class AssetRepository(Protocol):
    def create(self, asset: AssetRecord, version: AssetVersion) -> tuple[AssetRecord, AssetVersion]: ...

    def get(self, asset_id: str) -> AssetRecord | None: ...

    def version(self, asset_version_id: str) -> AssetVersion | None: ...

    def update_version(self, version: AssetVersion) -> AssetVersion: ...

    def list_versions(self, asset_id: str) -> list[AssetVersion]: ...


class ArtifactRepository(Protocol):
    def create(self, artifact: ArtifactRecord) -> ArtifactRecord: ...

    def get(self, artifact_id: str) -> ArtifactRecord | None: ...


class RunRepository(Protocol):
    def create(self, run: RunRecord, attempt: AttemptRecord, *, idempotency_key: str | None = None) -> RunRecord: ...

    def get(self, run_id: str) -> RunRecord | None: ...

    def update(self, run: RunRecord) -> RunRecord: ...

    def attempt(self, attempt_id: str) -> AttemptRecord | None: ...

    def list_attempts(self, run_id: str) -> list[AttemptRecord]: ...

    def create_attempt(self, attempt: AttemptRecord) -> AttemptRecord: ...

    def update_attempt(self, attempt: AttemptRecord) -> AttemptRecord: ...

    def by_idempotency_key(self, key: str) -> RunRecord | None: ...


class EventRepository(Protocol):
    def append(self, event: RunEvent) -> RunEvent: ...

    def list_after(self, run_id: str, attempt_id: str, after_seq: int = 0) -> list[RunEvent]: ...


class AuditRepository(Protocol):
    def append(self, event: AuditEvent) -> AuditEvent: ...

    def list_for_project(self, project_id: str) -> list[AuditEvent]: ...


class OutboxRepository(Protocol):
    def add(self, event: OutboxEvent) -> OutboxEvent: ...

    def pending(self, limit: int = 100) -> list[OutboxEvent]: ...

    def mark_published(self, event_id: str, published_at: str) -> OutboxEvent | None: ...


class UnitOfWork(Protocol):
    """Transaction boundary used by RunService.

    The in-memory implementation serializes the same operations with a lock;
    the PostgreSQL implementation can map this boundary to one transaction.
    """

    projects: ProjectRepository
    runs: RunRepository
    events: EventRepository
    audits: AuditRepository
    outbox: OutboxRepository
    assets: AssetRepository
    artifacts: ArtifactRepository

    def __enter__(self) -> "UnitOfWork": ...

    def __exit__(self, exc_type, exc_value, traceback) -> None: ...


class ObjectStore(Protocol):
    def put_file(self, key: str, path: Path, *, content_type: str | None = None) -> dict[str, str | int]: ...

    def presigned_get(self, key: str, *, expires_seconds: int = 900) -> str: ...

    def presigned_put(self, key: str, *, expires_seconds: int = 900, content_type: str | None = None) -> str: ...


class TaskDispatcher(Protocol):
    def enqueue(self, *, queue: str, task: str, payload: dict, idempotency_key: str) -> str: ...


__all__ = ["ArtifactRepository", "AssetRepository", "AuditRepository", "EventRepository", "MotionSourceAdapter", "ObjectStore", "OutboxRepository", "ProjectRepository", "RobotAdapter", "RunRepository", "Sim2SimAdapter", "TaskDispatcher", "TrainingBackendAdapter", "UnitOfWork"]
