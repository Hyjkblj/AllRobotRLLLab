"""PostgreSQL repositories implementing the domain UnitOfWork ports.

The application layer works with external string identities and immutable
Pydantic records. PostgreSQL uses UUID foreign keys, so ``users.external_id``
is the explicit identity mapping boundary rather than leaking UUID conversion
into domain services.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.app.domain.contracts import (
    AssetRecord,
    AssetVersion,
    ArtifactRecord,
    AuditEvent,
    AttemptRecord,
    OutboxEvent,
    P3RunState,
    ProjectMember,
    ProjectRecord,
    RunEvent,
    RunManifest,
    RunRecord,
)
from backend.app.domain.state_machine import RunStatus
from backend.app.infrastructure.memory import RepositoryConflict


def _uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"database identifiers must be UUIDs: {value!r}") from exc


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class _Repository:
    def __init__(self, connection) -> None:
        self.connection = connection

    def _user_uuid(self, external_id: str, email: str | None = None) -> uuid.UUID:
        with self.connection.cursor() as cursor:
            cursor.execute("select id from users where external_id = %s", (external_id,))
            row = cursor.fetchone()
            if row:
                return row[0]
            cursor.execute(
                "insert into users (id, external_id, email) values (gen_random_uuid(), %s, %s) returning id",
                (external_id, email),
            )
            return cursor.fetchone()[0]

    def _external_user(self, user_id: Any) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute("select external_id from users where id = %s", (user_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(f"user not found: {user_id}")
            return str(row[0])


class PostgresProjectRepository(_Repository):
    def create(self, project: ProjectRecord, owner: ProjectMember) -> ProjectRecord:
        if project.project_id != owner.project_id or project.owner_id != owner.user_id:
            raise ValueError("project owner membership does not match project")
        owner_uuid = self._user_uuid(owner.user_id)
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "insert into projects (id, name, owner_id, status, created_at, updated_at) values (%s,%s,%s,%s,%s,%s)",
                    (_uuid(project.project_id), project.name, owner_uuid, project.status.value, project.created_at, project.updated_at),
                )
                cursor.execute(
                    "insert into project_members (project_id, user_id, role, created_at) values (%s,%s,%s,%s)",
                    (_uuid(project.project_id), owner_uuid, owner.role.value, owner.created_at),
                )
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return project

    def get(self, project_id: str) -> ProjectRecord | None:
        try:
            project_uuid = _uuid(project_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select p.*, u.external_id as owner_external_id from projects p join users u on u.id=p.owner_id where p.id=%s", (project_uuid,))
            row = cursor.fetchone()
        return _project(row) if row else None

    def list_for_user(self, user_id: str) -> list[ProjectRecord]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select p.*, u.external_id as owner_external_id from projects p join project_members pm on pm.project_id=p.id join users m on m.id=pm.user_id join users u on u.id=p.owner_id where m.external_id=%s order by p.updated_at desc", (user_id,))
            return [_project(row) for row in cursor.fetchall()]

    def member(self, project_id: str, user_id: str) -> ProjectMember | None:
        try:
            project_uuid = _uuid(project_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select pm.*, u.external_id from project_members pm join users u on u.id=pm.user_id where pm.project_id=%s and u.external_id=%s", (project_uuid, user_id))
            row = cursor.fetchone()
        return _member(row, project_id) if row else None

    def list_members(self, project_id: str) -> list[ProjectMember]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select pm.*, u.external_id from project_members pm join users u on u.id=pm.user_id where pm.project_id=%s order by pm.created_at", (_uuid(project_id),))
            return [_member(row, project_id) for row in cursor.fetchall()]

    def add_member(self, member: ProjectMember) -> ProjectMember:
        user_uuid = self._user_uuid(member.user_id)
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into project_members (project_id,user_id,role,created_at) values (%s,%s,%s,%s)", (_uuid(member.project_id), user_uuid, member.role.value, member.created_at))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return member


class PostgresRunRepository(_Repository):
    def create(self, run: RunRecord, attempt: AttemptRecord, *, idempotency_key: str | None = None) -> RunRecord:
        if run.current_attempt_id != attempt.attempt_id or attempt.run_id != run.run_id:
            raise ValueError("run current attempt does not match attempt record")
        created_by = self._user_uuid(run.created_by)
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into runs (id,project_id,status,parent_run_id,created_by,current_attempt_id,manifest_json,manifest_sha256,created_at,updated_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(run.run_id), _uuid(run.project_id), run.status.value, _uuid(run.parent_run_id) if run.parent_run_id else None, created_by, _uuid(attempt.attempt_id), Jsonb(run.manifest.model_dump(mode="json")), run.manifest.manifest_sha256, run.created_at, run.updated_at))
                cursor.execute("insert into attempts (id,run_id,number,status,created_at,started_at,finished_at,worker_id,gpu_uuid,exit_code,failure_code,last_heartbeat_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(attempt.attempt_id), _uuid(attempt.run_id), attempt.number, attempt.status.value, attempt.created_at, attempt.started_at, attempt.finished_at, attempt.worker_id, attempt.gpu_uuid, attempt.exit_code, attempt.failure_code, attempt.last_heartbeat_at))
                if idempotency_key:
                    cursor.execute("insert into run_idempotency (idempotency_key,run_id) values (%s,%s)", (idempotency_key, _uuid(run.run_id)))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return run

    def get(self, run_id: str) -> RunRecord | None:
        try:
            run_uuid = _uuid(run_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select r.*, u.external_id as created_by_external_id from runs r join users u on u.id=r.created_by where r.id=%s", (run_uuid,))
            row = cursor.fetchone()
        return _run(row) if row else None

    def update(self, run: RunRecord) -> RunRecord:
        with self.connection.cursor() as cursor:
            cursor.execute("update runs set status=%s,parent_run_id=%s,created_by=%s,current_attempt_id=%s,manifest_json=%s,manifest_sha256=%s,updated_at=%s where id=%s", (run.status.value, _uuid(run.parent_run_id) if run.parent_run_id else None, self._user_uuid(run.created_by), _uuid(run.current_attempt_id), Jsonb(run.manifest.model_dump(mode="json")), run.manifest.manifest_sha256, run.updated_at, _uuid(run.run_id)))
            if cursor.rowcount != 1:
                raise KeyError(f"run not found: {run.run_id}")
        return run

    def attempt(self, attempt_id: str) -> AttemptRecord | None:
        try:
            attempt_uuid = _uuid(attempt_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from attempts where id=%s", (attempt_uuid,))
            row = cursor.fetchone()
        return _attempt(row) if row else None

    def list_attempts(self, run_id: str) -> list[AttemptRecord]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from attempts where run_id=%s order by number", (_uuid(run_id),))
            return [_attempt(row) for row in cursor.fetchall()]

    def create_attempt(self, attempt: AttemptRecord) -> AttemptRecord:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into attempts (id,run_id,number,status,created_at,started_at,finished_at,worker_id,gpu_uuid,exit_code,failure_code,last_heartbeat_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(attempt.attempt_id), _uuid(attempt.run_id), attempt.number, attempt.status.value, attempt.created_at, attempt.started_at, attempt.finished_at, attempt.worker_id, attempt.gpu_uuid, attempt.exit_code, attempt.failure_code, attempt.last_heartbeat_at))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return attempt

    def update_attempt(self, attempt: AttemptRecord) -> AttemptRecord:
        with self.connection.cursor() as cursor:
            cursor.execute("update attempts set run_id=%s,number=%s,status=%s,started_at=%s,finished_at=%s,worker_id=%s,gpu_uuid=%s,exit_code=%s,failure_code=%s,last_heartbeat_at=%s where id=%s", (_uuid(attempt.run_id), attempt.number, attempt.status.value, attempt.started_at, attempt.finished_at, attempt.worker_id, attempt.gpu_uuid, attempt.exit_code, attempt.failure_code, attempt.last_heartbeat_at, _uuid(attempt.attempt_id)))
            if cursor.rowcount != 1:
                raise KeyError(f"attempt not found: {attempt.attempt_id}")
        return attempt

    def by_idempotency_key(self, key: str) -> RunRecord | None:
        with self.connection.cursor() as cursor:
            cursor.execute("select run_id from run_idempotency where idempotency_key=%s", (key,))
            row = cursor.fetchone()
        return self.get(str(row[0])) if row else None


class PostgresEventRepository(_Repository):
    def append(self, event: RunEvent) -> RunEvent:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into log_events (run_id,attempt_id,seq,event_type,stage,level,message,payload_json,created_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(event.run_id), _uuid(event.attempt_id), event.seq, event.event_type, event.stage, event.level, event.message, Jsonb(event.payload), event.created_at))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return event

    def list_after(self, run_id: str, attempt_id: str, after_seq: int = 0) -> list[RunEvent]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from log_events where run_id=%s and attempt_id=%s and seq>%s order by seq", (_uuid(run_id), _uuid(attempt_id), after_seq))
            return [_event(row) for row in cursor.fetchall()]


class PostgresAuditRepository(_Repository):
    def append(self, event: AuditEvent) -> AuditEvent:
        with self.connection.cursor() as cursor:
            cursor.execute("insert into audit_events (id,project_id,actor_id,action,resource_type,resource_id,payload_json,created_at) values (%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(event.event_id), _uuid(event.project_id) if event.project_id else None, self._user_uuid(event.actor_id), event.action, event.resource_type, event.resource_id, Jsonb(event.payload), event.created_at))
        return event

    def list_for_project(self, project_id: str) -> list[AuditEvent]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select a.*,u.external_id as actor_external_id from audit_events a join users u on u.id=a.actor_id where a.project_id=%s order by a.created_at", (_uuid(project_id),))
            return [_audit(row) for row in cursor.fetchall()]


class PostgresOutboxRepository(_Repository):
    def add(self, event: OutboxEvent) -> OutboxEvent:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into outbox_events (id,topic,event_key,payload_json,created_at,published_at) values (%s,%s,%s,%s,%s,%s)", (_uuid(event.event_id), event.topic, event.key, Jsonb(event.payload), event.created_at, event.published_at))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return event

    def pending(self, limit: int = 100) -> list[OutboxEvent]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from outbox_events where published_at is null order by created_at limit %s", (limit,))
            return [_outbox(row) for row in cursor.fetchall()]

    def mark_published(self, event_id: str, published_at: str) -> OutboxEvent | None:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("update outbox_events set published_at=%s where id=%s returning *", (published_at, _uuid(event_id)))
            row = cursor.fetchone()
        return _outbox(row) if row else None


class PostgresAssetRepository(_Repository):
    def create(self, asset: AssetRecord, version: AssetVersion) -> tuple[AssetRecord, AssetVersion]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into assets (id,project_id,kind,display_name,license_json,status,created_by,created_at) values (%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(asset.asset_id), _uuid(asset.project_id), asset.kind.value, asset.display_name, Jsonb(asset.license.model_dump(mode="json")), asset.status.value, self._user_uuid(asset.created_by), asset.created_at))
                cursor.execute("insert into asset_versions (id,asset_id,version,status,object_key,original_filename,content_type,size_bytes,sha256,created_at,validated_at,rejection_code) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(version.asset_version_id), _uuid(version.asset_id), version.version, version.status.value, version.object_key, version.original_filename, version.content_type, version.size_bytes, version.sha256, version.created_at, version.validated_at, version.rejection_code))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return asset, version

    def get(self, asset_id: str) -> AssetRecord | None:
        try:
            asset_uuid = _uuid(asset_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select a.*,u.external_id as created_by_external_id from assets a join users u on u.id=a.created_by where a.id=%s", (asset_uuid,))
            row = cursor.fetchone()
        return _asset(row) if row else None

    def version(self, asset_version_id: str) -> AssetVersion | None:
        try:
            version_uuid = _uuid(asset_version_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from asset_versions where id=%s", (version_uuid,))
            row = cursor.fetchone()
        return _asset_version(row) if row else None

    def update_version(self, version: AssetVersion) -> AssetVersion:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("update asset_versions set status=%s,object_key=%s,original_filename=%s,content_type=%s,size_bytes=%s,sha256=%s,validated_at=%s,rejection_code=%s where id=%s", (version.status.value, version.object_key, version.original_filename, version.content_type, version.size_bytes, version.sha256, version.validated_at, version.rejection_code, _uuid(version.asset_version_id)))
                if cursor.rowcount != 1:
                    raise KeyError(f"asset version not found: {version.asset_version_id}")
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return version

    def list_versions(self, asset_id: str) -> list[AssetVersion]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from asset_versions where asset_id=%s order by version", (_uuid(asset_id),))
            return [_asset_version(row) for row in cursor.fetchall()]


class PostgresArtifactRepository(_Repository):
    def create(self, artifact: ArtifactRecord) -> ArtifactRecord:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("insert into artifacts (id,run_id,attempt_id,kind,object_key,sha256,size_bytes,content_type,created_at) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)", (_uuid(artifact.artifact_id), _uuid(artifact.run_id), _uuid(artifact.attempt_id), artifact.kind, artifact.object_key, artifact.sha256, artifact.size_bytes, artifact.content_type, artifact.created_at))
        except Exception as exc:
            if _is_integrity_error(exc):
                raise RepositoryConflict(str(exc)) from exc
            raise
        return artifact

    def get(self, artifact_id: str) -> ArtifactRecord | None:
        try:
            artifact_uuid = _uuid(artifact_id)
        except ValueError:
            return None
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from artifacts where id=%s", (artifact_uuid,))
            row = cursor.fetchone()
        return _artifact(row) if row else None

    def list_for_run(self, run_id: str) -> list[ArtifactRecord]:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from artifacts where run_id=%s order by created_at", (_uuid(run_id),))
            return [_artifact(row) for row in cursor.fetchall()]


class PostgresP3StateRepository(_Repository):
    def get(self, run_id: str) -> P3RunState | None:
        with self.connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute("select * from p3_run_states where run_id=%s", (_uuid(run_id),))
            row = cursor.fetchone()
        return _p3_state(row) if row else None

    def upsert(self, state: P3RunState) -> P3RunState:
        values = (
            _uuid(state.run_id),
            _uuid(state.attempt_id),
            Jsonb(state.training_config.model_dump(mode="json")) if state.training_config else None,
            Jsonb(state.checkpoint.model_dump(mode="json")) if state.checkpoint else None,
            Jsonb(state.export_metadata.model_dump(mode="json")) if state.export_metadata else None,
            Jsonb([item.model_dump(mode="json") for item in state.export_files]),
            Jsonb(state.bundle.model_dump(mode="json")) if state.bundle else None,
            Jsonb(state.sim2sim_report.model_dump(mode="json")) if state.sim2sim_report else None,
            Jsonb(state.artifact_ids),
            state.updated_at,
        )
        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                insert into p3_run_states
                  (run_id, attempt_id, training_config_json, checkpoint_json,
                   export_metadata_json, export_files_json, bundle_json,
                   sim2sim_report_json, artifact_ids_json, updated_at)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict (run_id) do update set
                  attempt_id=excluded.attempt_id,
                  training_config_json=excluded.training_config_json,
                  checkpoint_json=excluded.checkpoint_json,
                  export_metadata_json=excluded.export_metadata_json,
                  export_files_json=excluded.export_files_json,
                  bundle_json=excluded.bundle_json,
                  sim2sim_report_json=excluded.sim2sim_report_json,
                  artifact_ids_json=excluded.artifact_ids_json,
                  updated_at=excluded.updated_at
                returning *
                """,
                values,
            )
            row = cursor.fetchone()
        return _p3_state(row)


class PostgresUnitOfWork:
    """One connection/transaction per application service operation."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.connection = None

    def __enter__(self) -> "PostgresUnitOfWork":
        import psycopg

        self.connection = psycopg.connect(self.dsn)
        self.projects = PostgresProjectRepository(self.connection)
        self.runs = PostgresRunRepository(self.connection)
        self.events = PostgresEventRepository(self.connection)
        self.audits = PostgresAuditRepository(self.connection)
        self.outbox = PostgresOutboxRepository(self.connection)
        self.assets = PostgresAssetRepository(self.connection)
        self.artifacts = PostgresArtifactRepository(self.connection)
        self.p3_states = PostgresP3StateRepository(self.connection)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.connection is None:
            return
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()
            self.connection = None


def _is_integrity_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {"UniqueViolation", "ForeignKeyViolation", "CheckViolation", "NotNullViolation", "IntegrityError"}


def _project(row: dict[str, Any]) -> ProjectRecord:
    from backend.app.domain.contracts import ProjectStatus

    return ProjectRecord(project_id=str(row["id"]), name=row["name"], owner_id=row["owner_external_id"], status=ProjectStatus(row["status"]), created_at=_iso(row["created_at"]), updated_at=_iso(row["updated_at"]))


def _member(row: dict[str, Any], project_id: str) -> ProjectMember:
    from backend.app.domain.contracts import ProjectRole

    return ProjectMember(project_id=project_id, user_id=row["external_id"], role=ProjectRole(row["role"]), created_at=_iso(row["created_at"]))


def _run(row: dict[str, Any]) -> RunRecord:
    return RunRecord(run_id=str(row["id"]), project_id=str(row["project_id"]), status=RunStatus(row["status"]), created_by=row["created_by_external_id"], created_at=_iso(row["created_at"]), updated_at=_iso(row["updated_at"]), parent_run_id=str(row["parent_run_id"]) if row["parent_run_id"] else None, current_attempt_id=str(row["current_attempt_id"]), manifest=RunManifest.model_validate(row["manifest_json"]))


def _attempt(row: dict[str, Any]) -> AttemptRecord:
    return AttemptRecord(attempt_id=str(row["id"]), run_id=str(row["run_id"]), number=row["number"], status=RunStatus(row["status"]), created_at=_iso(row["created_at"]), started_at=_iso(row["started_at"]) if row["started_at"] else None, finished_at=_iso(row["finished_at"]) if row["finished_at"] else None, worker_id=row["worker_id"], gpu_uuid=row["gpu_uuid"], exit_code=row["exit_code"], failure_code=row["failure_code"], last_heartbeat_at=_iso(row["last_heartbeat_at"]) if row["last_heartbeat_at"] else None)


def _event(row: dict[str, Any]) -> RunEvent:
    return RunEvent(seq=row["seq"], run_id=str(row["run_id"]), attempt_id=str(row["attempt_id"]), event_type=row["event_type"], stage=row["stage"], level=row["level"], message=row["message"], payload=row["payload_json"], created_at=_iso(row["created_at"]))


def _audit(row: dict[str, Any]) -> AuditEvent:
    return AuditEvent(event_id=str(row["id"]), project_id=str(row["project_id"]) if row["project_id"] else None, actor_id=row["actor_external_id"], action=row["action"], resource_type=row["resource_type"], resource_id=row["resource_id"], payload=row["payload_json"], created_at=_iso(row["created_at"]))


def _outbox(row: dict[str, Any]) -> OutboxEvent:
    return OutboxEvent(event_id=str(row["id"]), topic=row["topic"], key=row["event_key"], payload=row["payload_json"], created_at=_iso(row["created_at"]), published_at=_iso(row["published_at"]) if row["published_at"] else None)


def _asset(row: dict[str, Any]) -> AssetRecord:
    from backend.app.domain.contracts import AssetKind, AssetStatus, LicenseInfo

    return AssetRecord(asset_id=str(row["id"]), project_id=str(row["project_id"]), kind=AssetKind(row["kind"]), display_name=row["display_name"], license=LicenseInfo.model_validate(row["license_json"]), status=AssetStatus(row["status"]), created_by=row["created_by_external_id"], created_at=_iso(row["created_at"]))


def _asset_version(row: dict[str, Any]) -> AssetVersion:
    from backend.app.domain.contracts import AssetVersionStatus

    return AssetVersion(asset_version_id=str(row["id"]), asset_id=str(row["asset_id"]), version=row["version"], status=AssetVersionStatus(row["status"]), object_key=row["object_key"], original_filename=row["original_filename"], content_type=row["content_type"], size_bytes=row["size_bytes"], sha256=row["sha256"], created_at=_iso(row["created_at"]), validated_at=_iso(row["validated_at"]) if row["validated_at"] else None, rejection_code=row["rejection_code"])


def _artifact(row: dict[str, Any]) -> ArtifactRecord:
    return ArtifactRecord(artifact_id=str(row["id"]), run_id=str(row["run_id"]), attempt_id=str(row["attempt_id"]), kind=row["kind"], object_key=row["object_key"], sha256=row["sha256"], size_bytes=row["size_bytes"], content_type=row["content_type"], created_at=_iso(row["created_at"]))


def _p3_state(row: dict[str, Any]) -> P3RunState:
    from backend.app.domain.contracts import CheckpointRecord, ExportFile, ExportMetadata, PolicyBundle, Sim2SimReport, TrainingConfig

    return P3RunState(
        run_id=str(row["run_id"]),
        attempt_id=str(row["attempt_id"]),
        training_config=TrainingConfig.model_validate(row["training_config_json"]) if row["training_config_json"] else None,
        checkpoint=CheckpointRecord.model_validate(row["checkpoint_json"]) if row["checkpoint_json"] else None,
        export_metadata=ExportMetadata.model_validate(row["export_metadata_json"]) if row["export_metadata_json"] else None,
        export_files=[ExportFile.model_validate(item) for item in (row["export_files_json"] or [])],
        bundle=PolicyBundle.model_validate(row["bundle_json"]) if row["bundle_json"] else None,
        sim2sim_report=Sim2SimReport.model_validate(row["sim2sim_report_json"]) if row["sim2sim_report_json"] else None,
        artifact_ids=list(row["artifact_ids_json"] or []),
        updated_at=_iso(row["updated_at"]),
    )


__all__ = ["PostgresUnitOfWork"]
