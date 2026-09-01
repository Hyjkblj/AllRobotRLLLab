"""Environment-backed settings without importing infrastructure clients."""

from __future__ import annotations

import os
import json
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.repository_root = Path(__file__).resolve().parents[3]
        self.motion_asset_root = Path(os.getenv("MOTION_ASSET_ROOT", str(self.repository_root))).resolve()
        self.app_env = os.getenv("APP_ENV", "development").strip().lower()
        requested_mode = os.getenv("ROBOTLAB_MODE", os.getenv("ROBOTLAB_STORAGE_MODE", "")).strip().lower()
        self.storage_mode = requested_mode or ("compose" if self.app_env in {"staging", "production"} else "local_file")
        if self.storage_mode not in {"local_file", "memory", "compose"}:
            raise ValueError("ROBOTLAB_MODE must be local_file, memory or compose")
        self.runtime_root = Path(os.getenv("ROBOTLAB_RUNTIME_DIR", str(self.repository_root / "runtime"))).expanduser().resolve()
        self.api_version = "v1"
        self.execution_mode = os.getenv("EXECUTION_MODE", "sync_smoke").strip().lower()
        self.p3_backend = os.getenv("P3_BACKEND", "fake_smoke").strip().lower()
        self.platform_role = os.getenv("PLATFORM_ROLE", "api").strip().lower()
        self.require_external_runtime = os.getenv("REQUIRE_EXTERNAL_RUNTIME", "").strip().lower() == "true" or self.platform_role in {"gpu", "worker-gpu"}
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        self.minio_bucket = os.getenv("MINIO_BUCKET", "allrobotrl")
        self.minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        self.worker_auth_token = os.getenv("WORKER_AUTH_TOKEN", "").strip()
        self.runtime_manifest_path = Path(os.getenv("RUNTIME_MANIFEST_PATH", str(self.repository_root / ".runtime" / "runtime-manifest.json"))).expanduser().resolve()
        registration_path = self.runtime_root / "runtime-registrations.json"
        try:
            registration_doc = json.loads(registration_path.read_text(encoding="utf-8")) if registration_path.is_file() else {}
            registration_doc = registration_doc if isinstance(registration_doc, dict) else {}
        except (OSError, ValueError, TypeError):
            registration_doc = {}
        def registered_path(name: str) -> str:
            item = registration_doc.get(name, {})
            return str(item.get("path", "")).strip() if isinstance(item, dict) else ""
        def registered_python(name: str) -> str:
            item = registration_doc.get(name, {})
            return str(item.get("python", "")).strip() if isinstance(item, dict) else ""
        # External runtimes intentionally have independent interpreters.  The
        # values below are paths/entry points only; importing their packages
        # remains the responsibility of the selected worker image.
        self.gmr_path = os.getenv("GMR_PATH", "").strip() or registered_path("gmr")
        self.gvhmr_path = os.getenv("GVHMR_PATH", "").strip() or registered_path("gvhmr")
        self.isaac_lab_path = os.getenv("ISAACLAB_PATH", "").strip() or registered_path("isaac_lab")
        self.isaac_sim_path = os.getenv("ISAACSIM_PATH", "").strip() or registered_path("isaac_sim")
        self.unitree_rl_lab_path = os.getenv("UNITREE_RL_LAB_PATH", "").strip() or registered_path("unitree_rl_lab")
        self.unitree_mujoco_path = os.getenv("UNITREE_MUJOCO_PATH", "").strip() or registered_path("unitree_mujoco")
        self.gmr_python = os.getenv("GMR_PYTHON", "").strip() or registered_python("gmr")
        self.gvhmr_python = os.getenv("GVHMR_PYTHON", "").strip() or registered_python("gvhmr")
        self.isaac_python = os.getenv("ISAAC_PYTHON", "").strip() or registered_python("isaac_lab")
        self.sim2sim_python = os.getenv("SIM2SIM_PYTHON", "").strip() or registered_python("unitree_mujoco")
        self.g1_mjcf_path = os.getenv("G1_MJCF_PATH", "").strip()

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
        if self.require_external_runtime and self.p3_backend in {"isaac_lab", "unitree_rl_lab"} and not self.isaac_lab_path:
            errors.append("ISAACLAB_PATH is required for the Isaac Lab P3 backend")
        if self.require_external_runtime and self.p3_backend in {"isaac_lab", "unitree_rl_lab"} and not self.unitree_rl_lab_path:
            errors.append("UNITREE_RL_LAB_PATH is required for the Unitree RL Lab training backend")
        if self.require_external_runtime and self.p3_backend in {"gmr_gvhmr", "isaac_lab", "unitree_rl_lab"} and not self.gmr_path:
            errors.append("GMR_PATH is required for motion retargeting")
        if self.require_external_runtime and self.p3_backend in {"gmr_gvhmr", "isaac_lab", "unitree_rl_lab"} and not self.gvhmr_path:
            errors.append("GVHMR_PATH is required for video motion processing")
        if self.require_external_runtime and self.p3_backend in {"unitree_mujoco", "isaac_lab", "unitree_rl_lab"} and not self.unitree_mujoco_path:
            errors.append("UNITREE_MUJOCO_PATH is required for the sim2sim P3 backend")
        return errors


settings = Settings()
