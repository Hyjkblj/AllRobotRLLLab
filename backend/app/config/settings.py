"""Environment-backed settings without importing infrastructure clients."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[3]
        self.motion_asset_root = Path(os.getenv("MOTION_ASSET_ROOT", str(self.repository_root))).resolve()
        self.app_env = os.getenv("APP_ENV", "development").strip().lower()
        self.api_version = "v1"
        self.execution_mode = os.getenv("EXECUTION_MODE", "sync_smoke").strip().lower()
        self.p3_backend = os.getenv("P3_BACKEND", "fake_smoke").strip().lower()
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "allrobotrl")
        self.minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.worker_auth_token = os.getenv("WORKER_AUTH_TOKEN", "").strip()
        self.runtime_manifest_path = Path(os.getenv("RUNTIME_MANIFEST_PATH", str(self.repository_root / ".runtime" / "runtime-manifest.json"))).expanduser().resolve()

    @property
    def is_deployed(self) -> bool:
        return self.app_env in {"staging", "production"}

    def deployment_errors(self) -> list[str]:
        """Return actionable errors for a staging/production process.

        Development and contract-test imports deliberately remain dependency
        free.  A deployed process, however, must fail fast instead of silently
        falling back to in-memory state or accepting unauthenticated worker
        writes.
        """
        if not self.is_deployed:
            return []
        errors: list[str] = []
        def placeholder(value: str | None) -> bool:
            text = (value or "").lower()
            return any(marker in text for marker in ("replace-with", "allrobotrl_dev_only", "changeme"))

        if not self.database_url:
            errors.append("DATABASE_URL is required when APP_ENV is staging/production")
        elif placeholder(self.database_url):
            errors.append("DATABASE_URL still contains a development placeholder")
        if not self.redis_url:
            errors.append("REDIS_URL is required when APP_ENV is staging/production")
        if not (self.minio_endpoint and self.minio_access_key and self.minio_secret_key):
            errors.append("MINIO_ENDPOINT, MINIO_ACCESS_KEY and MINIO_SECRET_KEY are required when APP_ENV is staging/production")
        elif placeholder(self.minio_access_key) or placeholder(self.minio_secret_key):
            errors.append("MinIO credentials still contain a development placeholder")
        if len(self.worker_auth_token) < 32:
            errors.append("WORKER_AUTH_TOKEN must be at least 32 characters when APP_ENV is staging/production")
        elif placeholder(self.worker_auth_token):
            errors.append("WORKER_AUTH_TOKEN still contains a development placeholder")
        if self.p3_backend == "fake_smoke":
            errors.append("P3_BACKEND must select a real Isaac/Unitree backend in staging/production; fake_smoke is development-only")
        return errors


settings = Settings()
