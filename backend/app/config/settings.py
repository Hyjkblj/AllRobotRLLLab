"""Environment-backed settings without importing infrastructure clients."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[3]
        self.motion_asset_root = Path(os.getenv("MOTION_ASSET_ROOT", str(self.repository_root))).resolve()
        self.api_version = "v1"
        self.execution_mode = os.getenv("EXECUTION_MODE", "sync_smoke").strip().lower()
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "allrobotrl")
        self.minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"


settings = Settings()
