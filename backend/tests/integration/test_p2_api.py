from fastapi.testclient import TestClient

from backend.app.main import app


def test_projects_assets_runs_and_sse() -> None:
    client = TestClient(app, headers={"X-User-Id": "p2-owner"})
    project_response = client.post("/api/v1/projects", json={"name": "P2 integration"})
    assert project_response.status_code == 200
    project_id = project_response.json()["item"]["project_id"]
    asset_response = client.post(f"/api/v1/projects/{project_id}/assets", json={"kind": "motion", "display_name": "wave", "original_filename": "wave.npz", "content_type": "application/octet-stream", "license": {"status": "declared", "source": "test"}})
    assert asset_response.status_code == 200
    version_id = asset_response.json()["version"]["asset_version_id"]
    upload_url = client.post(f"/api/v1/assets/{version_id}/upload-url")
    assert upload_url.status_code == 200
    assert upload_url.json()["upload"]["asset_version_id"] == version_id
    completed = client.post(f"/api/v1/assets/{version_id}/upload-complete", json={"sha256": "a" * 64, "size_bytes": 12})
    assert completed.status_code == 200
    assert completed.json()["item"]["status"] == "VALIDATING"
    run = client.post("/api/v1/runs", headers={"Idempotency-Key": "p2-run-1"}, json={"project_id": project_id, "robot": {"robot_id": "unitree_g1_29dof"}, "motion": {"train_motion_sha256": "a" * 64}, "reward_config_sha256": "b" * 64, "training_config_sha256": "c" * 64})
    assert run.status_code == 200
    run_id = run.json()["item"]["run_id"]
    events = client.get(f"/api/v1/runs/{run_id}/events")
    assert events.status_code == 200
    assert "event: status" in events.text
    log = client.post(f"/api/v1/runs/{run_id}/events", headers={"X-Worker-Id": "worker-1"}, json={"event_type": "log", "stage": "validate_input", "message": "validated", "payload": {"seq": 1}})
    assert log.status_code == 200
    resumed = client.get(f"/api/v1/runs/{run_id}/events", params={"after_seq": 1})
    assert "validated" in resumed.text
    cancelled = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["item"]["status"] == "CANCELLED"


def test_worker_status_and_heartbeat_require_worker_marker() -> None:
    client = TestClient(app, headers={"X-User-Id": "p2-worker-owner"})
    project = client.post("/api/v1/projects", json={"name": "P2 worker"}).json()["item"]["project_id"]
    run = client.post("/api/v1/runs", json={"project_id": project, "robot": {"robot_id": "unitree_g1_29dof"}, "motion": {"train_motion_sha256": "a" * 64}, "reward_config_sha256": "b" * 64, "training_config_sha256": "c" * 64}).json()["item"]["run_id"]
    assert client.post(f"/api/v1/runs/{run}/heartbeat", json={}).status_code == 403
    heartbeat = client.post(f"/api/v1/runs/{run}/heartbeat", headers={"X-Worker-Id": "worker-1"}, json={})
    assert heartbeat.status_code == 200
    status = client.post(f"/api/v1/runs/{run}/status", headers={"X-Worker-Id": "worker-1"}, json={"status": "VALIDATING", "stage": "validate_input"})
    assert status.status_code == 200
    assert status.json()["item"]["status"] == "VALIDATING"


def test_errors_use_top_level_contract_shape() -> None:
    client = TestClient(app, headers={"X-User-Id": "unknown-user"})
    response = client.get("/api/v1/projects/not-found")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "PROJECT_NOT_FOUND"
    assert response.json()["request_id"]


def test_infrastructure_health_does_not_claim_unconfigured_services_are_ready() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/health/infrastructure")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"
    assert response.json()["checks"]["postgres"]["state"] == "pending"
