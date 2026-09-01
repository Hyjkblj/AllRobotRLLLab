"""Run/attempt orchestration for the P2 backend boundary."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from backend.app.domain.contracts import (
    Actor,
    AttemptRecord,
    AuditEvent,
    LicenseInfo,
    OutboxEvent,
    ProjectMember,
    ProjectRecord,
    ProjectRole,
    RunEvent,
    RunManifest,
    RunRecord,
)
from backend.app.domain.state_machine import InvalidTransition, RunStatus, transition
from backend.app.infrastructure.memory import RepositoryConflict


class RunServiceError(ValueError):
    """Stable application error suitable for conversion to API error payloads."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunService:
    """Own run invariants while delegating storage and queue details to ports."""

    def __init__(self, uow) -> None:
        self.uow = uow

    def create_project(self, *, name: str, actor: Actor) -> ProjectRecord:
        project_id = str(uuid.uuid4())
        timestamp = utc_now()
        project = ProjectRecord(project_id=project_id, name=name, owner_id=actor.user_id, created_at=timestamp, updated_at=timestamp)
        owner = ProjectMember(project_id=project_id, user_id=actor.user_id, role=ProjectRole.OWNER, created_at=timestamp)
        try:
            with self.uow:
                self.uow.projects.create(project, owner)
                self._audit(actor, project_id, "project.created", "project", project_id, {"name": name})
        except RepositoryConflict as exc:
            raise RunServiceError("PROJECT_ALREADY_EXISTS", str(exc), status_code=409) from exc
        return project

    def get_project(self, *, project_id: str, actor: Actor) -> tuple[ProjectRecord, list[ProjectMember]]:
        with self.uow:
            project = self._project_or_error(project_id)
            self._require_member(project_id, actor.user_id, ProjectRole.VIEWER)
            return project, self.uow.projects.list_members(project_id)

    def list_projects(self, *, actor: Actor) -> list[ProjectRecord]:
        with self.uow:
            return self.uow.projects.list_for_user(actor.user_id)

    def list_runs(self, *, actor: Actor, project_id: str) -> list[RunRecord]:
        with self.uow:
            self._project_or_error(project_id)
            self._require_member(project_id, actor.user_id, ProjectRole.VIEWER)
            return self.uow.runs.list_for_project(project_id)

    def add_member(self, *, project_id: str, actor: Actor, user_id: str, role: ProjectRole) -> ProjectMember:
        with self.uow:
            self._project_or_error(project_id)
            self._require_member(project_id, actor.user_id, ProjectRole.OWNER)
            if role == ProjectRole.OWNER:
                raise RunServiceError("PROJECT_OWNER_TRANSFER_REQUIRED", "owner transfer must use a dedicated workflow")
            member = ProjectMember(project_id=project_id, user_id=user_id, role=role, created_at=utc_now())
            self.uow.projects.add_member(member)
            self._audit(actor, project_id, "project.member_added", "project_member", f"{project_id}:{user_id}", {"role": role.value})
            return member

    def create_run(
        self,
        *,
        actor: Actor,
        project_id: str,
        robot: dict[str, str],
        motion: dict[str, Any],
        reward_config_sha256: str,
        training_config_sha256: str,
        runtime: dict[str, str] | None = None,
        execution: dict[str, Any] | None = None,
        licenses: list[LicenseInfo] | None = None,
        parent_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[RunRecord, AttemptRecord, bool]:
        if idempotency_key:
            with self.uow:
                existing = self.uow.runs.by_idempotency_key(idempotency_key)
                if existing is not None:
                    if existing.project_id != project_id:
                        raise RunServiceError("RUN_IDEMPOTENCY_CONFLICT", "idempotency key is already bound to another project", status_code=409)
                    self._require_member(project_id, actor.user_id, ProjectRole.VIEWER)
                    attempts = self.uow.runs.list_attempts(existing.run_id)
                    return existing, attempts[0], True
        run_id = str(uuid.uuid4())
        attempt_id = str(uuid.uuid4())
        manifest = RunManifest(
            project_id=project_id,
            run_id=run_id,
            attempt_id=attempt_id,
            parent_run_id=parent_run_id,
            robot=robot,
            motion=motion,
            reward_config_sha256=reward_config_sha256,
            training_config_sha256=training_config_sha256,
            runtime=runtime or {},
            execution=execution or {},
            licenses=licenses or [],
        ).freeze()
        timestamp = utc_now()
        run = RunRecord(run_id=run_id, project_id=project_id, created_by=actor.user_id, created_at=timestamp, updated_at=timestamp, parent_run_id=parent_run_id, current_attempt_id=attempt_id, manifest=manifest)
        attempt = AttemptRecord(attempt_id=attempt_id, run_id=run_id, number=1, created_at=timestamp)
        with self.uow:
            self._project_or_error(project_id)
            self._require_member(project_id, actor.user_id, ProjectRole.EDITOR)
            if parent_run_id is not None:
                parent = self.uow.runs.get(parent_run_id)
                if parent is None:
                    raise RunServiceError("PARENT_RUN_NOT_FOUND", f"parent run not found: {parent_run_id}", status_code=404)
                if parent.project_id != project_id:
                    raise RunServiceError("PARENT_RUN_PROJECT_MISMATCH", "parent run belongs to another project", status_code=400)
            try:
                self.uow.runs.create(run, attempt, idempotency_key=idempotency_key)
            except RepositoryConflict as exc:
                if idempotency_key:
                    existing = self.uow.runs.by_idempotency_key(idempotency_key)
                    if existing is not None:
                        if existing.project_id != project_id:
                            raise RunServiceError("RUN_IDEMPOTENCY_CONFLICT", "idempotency key is already bound to another project", status_code=409)
                        return existing, self.uow.runs.list_attempts(existing.run_id)[0], True
                raise RunServiceError("RUN_IDEMPOTENCY_CONFLICT", str(exc), status_code=409) from exc
            self._emit(run, attempt, event_type="status", stage="run_create", message="Run created", payload={"status": run.status.value})
            self._audit(actor, project_id, "run.created", "run", run_id, {"attempt_id": attempt_id, "manifest_sha256": manifest.manifest_sha256})
            self.uow.outbox.add(OutboxEvent(event_id=str(uuid.uuid4()), topic="runs.created", key=attempt_id, payload={"run_id": run_id, "attempt_id": attempt_id, "stage": "validate_input"}, created_at=timestamp))
        return run, attempt, False

    def get_run(self, *, run_id: str, actor: Actor) -> tuple[RunRecord, list[AttemptRecord]]:
        with self.uow:
            run = self._run_or_error(run_id)
            self._require_member(run.project_id, actor.user_id, ProjectRole.VIEWER)
            return run, self.uow.runs.list_attempts(run_id)

    def cancel_run(self, *, run_id: str, actor: Actor) -> tuple[RunRecord, AttemptRecord]:
        with self.uow:
            run = self._run_or_error(run_id)
            self._require_member(run.project_id, actor.user_id, ProjectRole.EDITOR)
            try:
                next_status = transition(run.status, RunStatus.CANCELLED)
            except InvalidTransition as exc:
                raise RunServiceError("RUN_NOT_CANCELLABLE", str(exc), status_code=409) from exc
            attempt = self.uow.runs.attempt(run.current_attempt_id)
            if attempt is None:
                raise RunServiceError("ATTEMPT_NOT_FOUND", "current attempt not found", status_code=500)
            updated_attempt = attempt.model_copy(update={"status": RunStatus.CANCELLED, "finished_at": utc_now()})
            updated_run = run.model_copy(update={"status": next_status, "updated_at": utc_now()})
            self.uow.runs.update_attempt(updated_attempt)
            self.uow.runs.update(updated_run)
            self._emit(updated_run, updated_attempt, event_type="status", stage="cancel", message="Run cancelled", payload={"status": next_status.value})
            self._audit(actor, run.project_id, "run.cancelled", "run", run_id, {"attempt_id": attempt.attempt_id})
            self.uow.outbox.add(OutboxEvent(event_id=str(uuid.uuid4()), topic="runs.cancelled", key=attempt.attempt_id, payload={"run_id": run_id, "attempt_id": attempt.attempt_id}, created_at=utc_now()))
            return updated_run, updated_attempt

    def retry_run(self, *, run_id: str, actor: Actor) -> tuple[RunRecord, AttemptRecord]:
        with self.uow:
            run = self._run_or_error(run_id)
            self._require_member(run.project_id, actor.user_id, ProjectRole.EDITOR)
            if run.status not in {RunStatus.FAILED, RunStatus.CANCELLED}:
                raise RunServiceError("RUN_NOT_RETRYABLE", "only FAILED or CANCELLED runs can be retried", status_code=409)
            if run.status == RunStatus.CANCELLED:
                # Cancellation is terminal for the current attempt, but the
                # immutable manifest can safely start a fresh attempt.
                retry_status = RunStatus.TRAINING_PREPARING
            else:
                try:
                    retry_status = transition(run.status, RunStatus.TRAINING_PREPARING)
                except InvalidTransition as exc:
                    raise RunServiceError("RUN_NOT_RETRYABLE", str(exc), status_code=409) from exc
            attempts = self.uow.runs.list_attempts(run_id)
            timestamp = utc_now()
            attempt = AttemptRecord(attempt_id=str(uuid.uuid4()), run_id=run_id, number=len(attempts) + 1, status=retry_status, created_at=timestamp)
            updated_manifest = run.manifest.model_copy(update={"attempt_id": attempt.attempt_id, "manifest_sha256": None}).freeze()
            updated_run = run.model_copy(update={"status": retry_status, "current_attempt_id": attempt.attempt_id, "manifest": updated_manifest, "updated_at": timestamp})
            self.uow.runs.create_attempt(attempt)
            self.uow.runs.update(updated_run)
            self._emit(updated_run, attempt, event_type="status", stage="retry", message="Retry attempt created", payload={"status": retry_status.value, "attempt_number": attempt.number})
            self._audit(actor, run.project_id, "run.retried", "run", run_id, {"attempt_id": attempt.attempt_id, "attempt_number": attempt.number})
            retry_payload = {"run_id": run_id, "attempt_id": attempt.attempt_id, "stage": "training_prepare"}
            prior_state = self.uow.p3_states.get(run_id)
            if prior_state is not None and prior_state.training_config is not None:
                retry_payload["config"] = prior_state.training_config.model_dump(mode="json")
            self.uow.outbox.add(OutboxEvent(event_id=str(uuid.uuid4()), topic="runs.retry", key=attempt.attempt_id, payload=retry_payload, created_at=timestamp))
            return updated_run, attempt

    def transition_run(self, *, run_id: str, target: RunStatus, actor: Actor | None = None, stage: str | None = None, message: str = "") -> RunRecord:
        with self.uow:
            run = self._run_or_error(run_id)
            if actor is not None:
                self._require_member(run.project_id, actor.user_id, ProjectRole.EDITOR)
            try:
                status = transition(run.status, target)
            except InvalidTransition as exc:
                raise RunServiceError("RUN_INVALID_TRANSITION", str(exc), status_code=409) from exc
            attempt = self.uow.runs.attempt(run.current_attempt_id)
            if attempt is None:
                raise RunServiceError("ATTEMPT_NOT_FOUND", "current attempt not found", status_code=500)
            updated_run = run.model_copy(update={"status": status, "updated_at": utc_now()})
            updated_attempt = attempt.model_copy(update={"status": status})
            self.uow.runs.update(updated_run)
            self.uow.runs.update_attempt(updated_attempt)
            self._emit(updated_run, updated_attempt, event_type="status", stage=stage, message=message or status.value, payload={"status": status.value})
            return updated_run

    def heartbeat(self, *, run_id: str, worker_id: str, gpu_uuid: str | None = None) -> AttemptRecord:
        if not worker_id.strip():
            raise RunServiceError("WORKER_ID_REQUIRED", "worker_id is required")
        with self.uow:
            run = self._run_or_error(run_id)
            attempt = self.uow.runs.attempt(run.current_attempt_id)
            if attempt is None:
                raise RunServiceError("ATTEMPT_NOT_FOUND", "current attempt not found", status_code=500)
            updated = attempt.model_copy(update={"worker_id": worker_id, "gpu_uuid": gpu_uuid, "last_heartbeat_at": utc_now()})
            self.uow.runs.update_attempt(updated)
            return updated

    def append_event(self, *, run_id: str, event_type: str, stage: str | None = None, level: str = "INFO", message: str = "", payload: dict[str, Any] | None = None) -> RunEvent:
        with self.uow:
            run = self._run_or_error(run_id)
            attempt = self.uow.runs.attempt(run.current_attempt_id)
            if attempt is None:
                raise RunServiceError("ATTEMPT_NOT_FOUND", "current attempt not found", status_code=500)
            if event_type not in {"status", "log", "metric", "system"}:
                raise RunServiceError("EVENT_TYPE_INVALID", f"unsupported run event type: {event_type}")
            return self._emit(run, attempt, event_type=event_type, stage=stage, level=level, message=message, payload=payload or {})

    def list_events(self, *, run_id: str, actor: Actor, after_seq: int = 0, attempt_id: str | None = None) -> list[RunEvent]:
        with self.uow:
            run = self._run_or_error(run_id)
            self._require_member(run.project_id, actor.user_id, ProjectRole.VIEWER)
            selected_attempt_id = attempt_id or run.current_attempt_id
            attempt = self.uow.runs.attempt(selected_attempt_id)
            if attempt is None or attempt.run_id != run_id:
                raise RunServiceError("ATTEMPT_NOT_FOUND", "requested attempt does not belong to this run", status_code=404)
            return self.uow.events.list_after(run_id, selected_attempt_id, after_seq)

    def _project_or_error(self, project_id: str) -> ProjectRecord:
        project = self.uow.projects.get(project_id)
        if project is None:
            raise RunServiceError("PROJECT_NOT_FOUND", f"project not found: {project_id}", status_code=404)
        return project

    def _run_or_error(self, run_id: str) -> RunRecord:
        run = self.uow.runs.get(run_id)
        if run is None:
            raise RunServiceError("RUN_NOT_FOUND", f"run not found: {run_id}", status_code=404)
        return run

    def _require_member(self, project_id: str, user_id: str, minimum: ProjectRole) -> ProjectMember:
        member = self.uow.projects.member(project_id, user_id)
        if member is None:
            raise RunServiceError("PROJECT_ACCESS_DENIED", "user is not a member of this project", status_code=403)
        hierarchy = {ProjectRole.VIEWER: 1, ProjectRole.EDITOR: 2, ProjectRole.OWNER: 3}
        if hierarchy[member.role] < hierarchy[minimum]:
            raise RunServiceError("PROJECT_ACCESS_DENIED", f"role {member.role.value} cannot perform this action", status_code=403)
        return member

    def _emit(self, run: RunRecord, attempt: AttemptRecord, *, event_type: str, stage: str | None, level: str = "INFO", message: str, payload: dict[str, Any]) -> RunEvent:
        previous = self.uow.events.list_after(run.run_id, attempt.attempt_id, 0)
        event = RunEvent(seq=len(previous) + 1, run_id=run.run_id, attempt_id=attempt.attempt_id, event_type=event_type, stage=stage, level=level, message=message, payload=payload, created_at=utc_now())
        return self.uow.events.append(event)

    def _audit(self, actor: Actor, project_id: str | None, action: str, resource_type: str, resource_id: str, payload: dict[str, Any]) -> AuditEvent:
        event = AuditEvent(event_id=str(uuid.uuid4()), project_id=project_id, actor_id=actor.user_id, action=action, resource_type=resource_type, resource_id=resource_id, payload=payload, created_at=utc_now())
        return self.uow.audits.append(event)


__all__ = ["RunService", "RunServiceError", "utc_now"]
