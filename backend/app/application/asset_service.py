"""Asset/version metadata orchestration and object-store upload sessions."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath

from backend.app.domain.contracts import Actor, AssetKind, AssetRecord, AssetVersion, AssetVersionStatus, AuditEvent, LicenseInfo, OutboxEvent, ProjectRole, UploadSession
from backend.app.infrastructure.memory import RepositoryConflict
from .run_service import RunServiceError, utc_now


class AssetService:
    def __init__(self, uow, object_store) -> None:
        self.uow = uow
        self.object_store = object_store

    def create_asset(self, *, actor: Actor, project_id: str, kind: AssetKind, display_name: str, original_filename: str, license: LicenseInfo, content_type: str | None = None) -> tuple[AssetRecord, AssetVersion, UploadSession]:
        with self.uow:
            self._require_member(project_id, actor.user_id, ProjectRole.EDITOR)
            asset_id = str(uuid.uuid4())
            version_id = str(uuid.uuid4())
            safe_filename = PurePosixPath(original_filename.replace("\\", "/")).name
            if not safe_filename or safe_filename in {".", ".."}:
                raise RunServiceError("ASSET_FILENAME_INVALID", "original_filename must contain a file name")
            suffix = PurePosixPath(safe_filename).suffix.lower()
            suffix = suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ""
            object_key = f"projects/{project_id}/assets/{asset_id}/versions/{version_id}/source/{uuid.uuid4().hex}{suffix}"
            timestamp = utc_now()
            asset = AssetRecord(asset_id=asset_id, project_id=project_id, kind=kind, display_name=display_name, license=license, created_by=actor.user_id, created_at=timestamp)
            version = AssetVersion(asset_version_id=version_id, asset_id=asset_id, version=1, object_key=object_key, original_filename=safe_filename, content_type=content_type, created_at=timestamp)
            try:
                self.uow.assets.create(asset, version)
            except RepositoryConflict as exc:
                raise RunServiceError("ASSET_ALREADY_EXISTS", str(exc), status_code=409) from exc
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
            session = UploadSession(session_id=str(uuid.uuid4()), asset_version_id=version_id, object_key=object_key, upload_url=self.object_store.presigned_put(object_key, content_type=content_type), expires_at=expires_at)
            self.uow.audits.append(AuditEvent(event_id=str(uuid.uuid4()), project_id=project_id, actor_id=actor.user_id, action="asset.created", resource_type="asset", resource_id=asset_id, payload={"asset_version_id": version_id}, created_at=timestamp))
            self.uow.outbox.add(OutboxEvent(event_id=str(uuid.uuid4()), topic="assets.uploading", key=version_id, payload={"asset_id": asset_id, "asset_version_id": version_id, "object_key": object_key}, created_at=timestamp))
            return asset, version, session

    def complete_upload(self, *, actor: Actor, asset_version_id: str, sha256: str | None, size_bytes: int | None) -> AssetVersion:
        with self.uow:
            version = self.uow.assets.version(asset_version_id)
            if version is None:
                raise RunServiceError("ASSET_VERSION_NOT_FOUND", f"asset version not found: {asset_version_id}", status_code=404)
            asset = self.uow.assets.get(version.asset_id)
            if asset is None:
                raise RunServiceError("ASSET_NOT_FOUND", f"asset not found: {version.asset_id}", status_code=404)
            self._require_member(asset.project_id, actor.user_id, ProjectRole.EDITOR)
            if version.status != AssetVersionStatus.UPLOADING:
                raise RunServiceError("ASSET_UPLOAD_NOT_OPEN", "asset version is not accepting upload completion", status_code=409)
            if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
                raise RunServiceError("ASSET_HASH_INVALID", "sha256 must be a lowercase 64-character hex digest")
            computed_sha256 = sha256
            computed_size = size_bytes
            stat = getattr(self.object_store, "stat", None)
            if callable(stat):
                try:
                    metadata = stat(version.object_key)
                except Exception:
                    metadata = None
                if metadata is not None:
                    server_sha256 = str(metadata.get("sha256") or "")
                    if sha256 is not None and server_sha256 and server_sha256 != sha256:
                        raise RunServiceError("ASSET_HASH_MISMATCH", "server-computed object hash differs from the client declaration")
                    if size_bytes is not None and int(metadata["size_bytes"]) != size_bytes:
                        raise RunServiceError("ASSET_SIZE_MISMATCH", "server-computed object size differs from the client declaration")
                    computed_sha256 = server_sha256 or computed_sha256
                    computed_size = int(metadata["size_bytes"])
            updated = version.model_copy(update={"status": AssetVersionStatus.VALIDATING, "sha256": computed_sha256, "size_bytes": computed_size})
            try:
                self.uow.assets.update_version(updated)
            except RepositoryConflict as exc:
                raise RunServiceError("ASSET_HASH_ALREADY_EXISTS", "asset sha256 is already registered", status_code=409) from exc
            timestamp = utc_now()
            self.uow.audits.append(AuditEvent(event_id=str(uuid.uuid4()), project_id=asset.project_id, actor_id=actor.user_id, action="asset.upload_completed", resource_type="asset_version", resource_id=asset_version_id, payload={"sha256": sha256, "size_bytes": size_bytes}, created_at=timestamp))
            self.uow.outbox.add(OutboxEvent(event_id=str(uuid.uuid4()), topic="assets.validate", key=asset_version_id, payload={"asset_id": asset.asset_id, "asset_version_id": asset_version_id, "object_key": version.object_key}, created_at=timestamp))
            return updated

    def upload_session(self, *, actor: Actor, asset_version_id: str) -> UploadSession:
        with self.uow:
            version = self.uow.assets.version(asset_version_id)
            if version is None:
                raise RunServiceError("ASSET_VERSION_NOT_FOUND", f"asset version not found: {asset_version_id}", status_code=404)
            asset = self.uow.assets.get(version.asset_id)
            if asset is None:
                raise RunServiceError("ASSET_NOT_FOUND", f"asset not found: {version.asset_id}", status_code=404)
            self._require_member(asset.project_id, actor.user_id, ProjectRole.EDITOR)
            if version.status != AssetVersionStatus.UPLOADING:
                raise RunServiceError("ASSET_UPLOAD_NOT_OPEN", "asset version is not accepting uploads", status_code=409)
            return UploadSession(session_id=str(uuid.uuid4()), asset_version_id=asset_version_id, object_key=version.object_key, upload_url=self.object_store.presigned_put(version.object_key, content_type=version.content_type), expires_at=(datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat())

    def mark_validated(self, *, asset_version_id: str, valid: bool, sha256: str | None = None, size_bytes: int | None = None, rejection_code: str | None = None) -> AssetVersion:
        with self.uow:
            version = self.uow.assets.version(asset_version_id)
            if version is None:
                raise RunServiceError("ASSET_VERSION_NOT_FOUND", f"asset version not found: {asset_version_id}", status_code=404)
            status = AssetVersionStatus.READY if valid else AssetVersionStatus.REJECTED
            updated = version.model_copy(update={"status": status, "sha256": sha256 or version.sha256, "size_bytes": size_bytes if size_bytes is not None else version.size_bytes, "rejection_code": None if valid else rejection_code, "validated_at": utc_now()})
            try:
                return self.uow.assets.update_version(updated)
            except RepositoryConflict as exc:
                raise RunServiceError("ASSET_HASH_ALREADY_EXISTS", "asset sha256 is already registered", status_code=409) from exc

    def list_versions(self, *, actor: Actor, asset_id: str) -> tuple[AssetRecord, list[AssetVersion]]:
        with self.uow:
            asset = self.uow.assets.get(asset_id)
            if asset is None:
                raise RunServiceError("ASSET_NOT_FOUND", f"asset not found: {asset_id}", status_code=404)
            self._require_member(asset.project_id, actor.user_id, ProjectRole.VIEWER)
            return asset, self.uow.assets.list_versions(asset_id)

    def _require_member(self, project_id: str, user_id: str, minimum: ProjectRole) -> None:
        member = self.uow.projects.member(project_id, user_id)
        if member is None:
            raise RunServiceError("PROJECT_ACCESS_DENIED", "user is not a member of this project", status_code=403)
        hierarchy = {ProjectRole.VIEWER: 1, ProjectRole.EDITOR: 2, ProjectRole.OWNER: 3}
        if hierarchy[member.role] < hierarchy[minimum]:
            raise RunServiceError("PROJECT_ACCESS_DENIED", "project role cannot perform this action", status_code=403)


__all__ = ["AssetService"]
