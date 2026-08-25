"""Artifact registration and permission-checked download URLs."""

from __future__ import annotations

import uuid

from backend.app.domain.contracts import Actor, ArtifactRecord, ProjectRole
from backend.app.infrastructure.memory import RepositoryConflict
from backend.app.infrastructure.object_store import validate_object_key
from .run_service import RunServiceError, utc_now


class ArtifactService:
    def __init__(self, uow, object_store) -> None:
        self.uow = uow
        self.object_store = object_store

    def register(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self.uow:
            run = self.uow.runs.get(artifact.run_id)
            attempt = self.uow.runs.attempt(artifact.attempt_id)
            if run is None or attempt is None or attempt.run_id != run.run_id:
                raise RunServiceError("ARTIFACT_ATTEMPT_INVALID", "artifact run/attempt relationship is invalid")
            try:
                validate_object_key(artifact.object_key)
            except ValueError as exc:
                raise RunServiceError("ARTIFACT_OBJECT_KEY_INVALID", str(exc)) from exc
            try:
                return self.uow.artifacts.create(artifact)
            except RepositoryConflict as exc:
                raise RunServiceError("ARTIFACT_ALREADY_EXISTS", str(exc), status_code=409) from exc

    def get_download(self, *, artifact_id: str, actor: Actor) -> tuple[ArtifactRecord, str]:
        with self.uow:
            artifact = self.uow.artifacts.get(artifact_id)
            if artifact is None:
                raise RunServiceError("ARTIFACT_NOT_FOUND", f"artifact not found: {artifact_id}", status_code=404)
            run = self.uow.runs.get(artifact.run_id)
            if run is None:
                raise RunServiceError("RUN_NOT_FOUND", f"run not found: {artifact.run_id}", status_code=404)
            member = self.uow.projects.member(run.project_id, actor.user_id)
            if member is None:
                raise RunServiceError("PROJECT_ACCESS_DENIED", "user is not a member of this project", status_code=403)
            try:
                url = self.object_store.presigned_get(artifact.object_key)
            except Exception as exc:
                raise RunServiceError("ARTIFACT_OBJECT_UNAVAILABLE", str(exc), status_code=503) from exc
            return artifact, url

    def list_for_run(self, *, run_id: str, actor: Actor) -> list[ArtifactRecord]:
        with self.uow:
            run = self.uow.runs.get(run_id)
            if run is None:
                raise RunServiceError("RUN_NOT_FOUND", f"run not found: {run_id}", status_code=404)
            if self.uow.projects.member(run.project_id, actor.user_id) is None:
                raise RunServiceError("PROJECT_ACCESS_DENIED", "user is not a member of this project", status_code=403)
            return self.uow.artifacts.list_for_run(run_id)


__all__ = ["ArtifactService"]
