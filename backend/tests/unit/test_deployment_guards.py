from fastapi.testclient import TestClient

from backend.app.config.settings import settings
from backend.app.main import app
from backend.app.workers.p3_tasks import P3TaskExecutor


def test_deployed_settings_require_durable_services_and_worker_token() -> None:
    original = (settings.app_env, settings.database_url, settings.redis_url, settings.minio_endpoint, settings.minio_access_key, settings.minio_secret_key, settings.worker_auth_token)
    try:
        settings.app_env = "staging"
        settings.database_url = None
        settings.redis_url = None
        settings.minio_endpoint = None
        settings.minio_access_key = None
        settings.minio_secret_key = None
        settings.worker_auth_token = "short"
        errors = settings.deployment_errors()
        assert any("DATABASE_URL" in error for error in errors)
        assert any("WORKER_AUTH_TOKEN" in error for error in errors)
    finally:
        (settings.app_env, settings.database_url, settings.redis_url, settings.minio_endpoint, settings.minio_access_key, settings.minio_secret_key, settings.worker_auth_token) = original


def test_worker_token_is_checked_when_configured() -> None:
    original = settings.worker_auth_token
    try:
        settings.worker_auth_token = "t" * 32
        response = TestClient(app).post("/api/v1/runs/missing/train", headers={"X-Worker-Id": "worker"}, json={"motion_asset_version_id": "motion-1"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "WORKER_AUTH_INVALID"
    finally:
        settings.worker_auth_token = original


def test_deployed_worker_never_falls_back_to_cpu_smoke(tmp_path) -> None:
    original = (settings.app_env, settings.p3_backend)
    try:
        settings.app_env = "staging"
        settings.p3_backend = "isaac_lab"
        try:
            P3TaskExecutor(None).execute({"operation": "train", "run_id": "run-1"})
        except RuntimeError as exc:
            assert "refusing to run CPU smoke" in str(exc)
        else:
            raise AssertionError("deployed worker accepted the CPU smoke fallback")
    finally:
        settings.app_env, settings.p3_backend = original
