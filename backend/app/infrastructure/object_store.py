"""S3-shaped object-store adapters for local development.

The local implementation is intentionally explicit and recoverable; production
can replace it with MinIO/S3 without changing application services.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote


_SAFE_KEY = re.compile(r"^[A-Za-z0-9._/-]+$")


class ObjectStoreError(ValueError):
    pass


def validate_object_key(key: str) -> str:
    normalized = key.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or ".." in normalized.split("/") or not _SAFE_KEY.fullmatch(normalized):
        raise ObjectStoreError("object key contains an unsafe path")
    return normalized


class LocalObjectStore:
    """Filesystem-backed object store for Local File Mode.

    ``content_addressed=True`` stores bytes below ``sha256/<digest>/`` and
    keeps a small key index so API object keys remain compatible with MinIO.
    """

    def __init__(self, root: Path, *, content_addressed: bool = False) -> None:
        self.root = root.resolve()
        self.content_addressed = content_addressed
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"

    def put_file(self, key: str, path: Path, *, content_type: str | None = None) -> dict[str, str | int]:
        safe_key = validate_object_key(key)
        source = path.resolve()
        if not source.is_file():
            raise ObjectStoreError(f"source file does not exist: {source}")
        digest_builder = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        target = self._target_for_key(safe_key, digest)
        target.relative_to(self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".object.", suffix=".tmp", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream, source.open("rb") as input_stream:
                shutil.copyfileobj(input_stream, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        if self.content_addressed:
            index = self._read_index()
            index[safe_key] = str(target.relative_to(self.root)).replace(os.sep, "/")
            self._write_index(index)
        return {"key": safe_key, "sha256": digest, "size_bytes": target.stat().st_size, "content_type": content_type or "application/octet-stream"}

    def presigned_get(self, key: str, *, expires_seconds: int = 900) -> str:
        safe_key = validate_object_key(key)
        if not self._target_for_key(safe_key).is_file():
            raise ObjectStoreError(f"object not found: {safe_key}")
        return f"/objects/{quote(safe_key, safe='/')}?expires={int(expires_seconds)}"

    def presigned_put(self, key: str, *, expires_seconds: int = 900, content_type: str | None = None) -> str:
        safe_key = validate_object_key(key)
        return f"/uploads/{quote(safe_key, safe='/')}?expires={int(expires_seconds)}"

    def stat(self, key: str) -> dict[str, str | int]:
        safe_key = validate_object_key(key)
        target = self._target_for_key(safe_key).resolve()
        target.relative_to(self.root)
        if not target.is_file():
            raise ObjectStoreError(f"object not found: {safe_key}")
        return {"key": safe_key, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "size_bytes": target.stat().st_size}

    def _target_for_key(self, safe_key: str, digest: str | None = None) -> Path:
        if self.content_addressed:
            relative = self._read_index().get(safe_key)
            if relative:
                return (self.root / relative).resolve()
            if digest:
                return (self.root / "sha256" / digest / safe_key).resolve()
        return (self.root / safe_key).resolve()

    def _read_index(self) -> dict[str, str]:
        if not self._index_path.exists():
            return {}
        try:
            value = json.loads(self._index_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _write_index(self, value: dict[str, str]) -> None:
        temporary = self._index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self._index_path)


class MinioObjectStore:
    """Production S3-compatible adapter backed by the MinIO client."""

    def __init__(self, *, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = True) -> None:
        from minio import Minio

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket

    def put_file(self, key: str, path: Path, *, content_type: str | None = None) -> dict[str, str | int]:
        safe_key = validate_object_key(key)
        source = path.resolve()
        if not source.is_file():
            raise ObjectStoreError(f"source file does not exist: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        self.client.fput_object(self.bucket, safe_key, str(source), content_type=content_type or "application/octet-stream")
        return {"key": safe_key, "sha256": digest.hexdigest(), "size_bytes": source.stat().st_size, "content_type": content_type or "application/octet-stream"}

    def presigned_get(self, key: str, *, expires_seconds: int = 900) -> str:
        return self.client.presigned_get_object(self.bucket, validate_object_key(key), expires=timedelta(seconds=expires_seconds))

    def presigned_put(self, key: str, *, expires_seconds: int = 900, content_type: str | None = None) -> str:
        return self.client.presigned_put_object(self.bucket, validate_object_key(key), expires=timedelta(seconds=expires_seconds))

    def stat(self, key: str) -> dict[str, str | int]:
        safe_key = validate_object_key(key)
        metadata = self.client.stat_object(self.bucket, safe_key)
        # S3 ETags are not SHA-256 for multipart uploads. Server-side hash is
        # completed by the asset validation worker and then persisted.
        return {"key": safe_key, "size_bytes": int(metadata.size), "sha256": str(metadata.metadata.get("X-Amz-Meta-Sha256", ""))}


__all__ = ["LocalObjectStore", "MinioObjectStore", "ObjectStoreError", "validate_object_key"]
