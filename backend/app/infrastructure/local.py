"""Infrastructure factories shared by API and worker processes."""

from __future__ import annotations

from backend.app.infrastructure.object_store import LocalObjectStore, MinioObjectStore


def build_object_store(settings):
    if settings.minio_endpoint and settings.minio_access_key and settings.minio_secret_key:
        return MinioObjectStore(endpoint=settings.minio_endpoint, access_key=settings.minio_access_key, secret_key=settings.minio_secret_key, bucket=settings.minio_bucket, secure=settings.minio_secure)
    return LocalObjectStore(settings.runtime_root / "artifacts", content_addressed=True)


__all__ = ["build_object_store"]
