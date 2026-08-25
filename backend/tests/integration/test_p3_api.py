from fastapi.testclient import TestClient

from backend.app.main import app


def test_p3_train_export_sim2sim_api_smoke() -> None:
    client = TestClient(app, headers={"X-User-Id": "p3-owner"})
    project = client.post("/api/v1/projects", json={"name": "P3 smoke"}).json()["item"]["project_id"]
    run_response = client.post("/api/v1/runs", json={"project_id": project, "robot": {"robot_id": "unitree_g1_29dof"}, "motion": {"train_motion_sha256": "a" * 64}, "reward_config_sha256": "b" * 64, "training_config_sha256": "c" * 64})
    assert run_response.status_code == 200
    run_id = run_response.json()["item"]["run_id"]
    train = client.post(f"/api/v1/runs/{run_id}/train", headers={"X-Worker-Id": "p3-worker"}, json={"motion_asset_version_id": "motion-1"})
    assert train.status_code == 200, train.text
    export = client.post(f"/api/v1/runs/{run_id}/export", headers={"X-Worker-Id": "p3-worker"})
    assert export.status_code == 200, export.text
    assert export.json()["item"]["export"]["smoke_passed"] is True
    sim2sim = client.post(f"/api/v1/runs/{run_id}/sim2sim", headers={"X-Worker-Id": "p3-worker"}, json={})
    assert sim2sim.status_code == 200, sim2sim.text
    assert sim2sim.json()["item"]["status"] == "PASSED"
    run = client.get(f"/api/v1/runs/{run_id}")
    assert run.json()["item"]["status"] == "READY_TO_DOWNLOAD"
    artifacts = client.get(f"/api/v1/runs/{run_id}/artifacts")
    assert artifacts.status_code == 200, artifacts.text
    assert {item["kind"] for item in artifacts.json()["items"]} >= {"checkpoint", "policy_bundle", "sim2sim_report", "policy_bundle_final"}
    artifact = client.get(f"/api/v1/artifacts/{artifacts.json()['items'][0]['artifact_id']}")
    assert artifact.status_code == 200, artifact.text
    assert artifact.json()["download_url"]


def test_p3_async_train_submission_returns_202() -> None:
    client = TestClient(app, headers={"X-User-Id": "p3-async-owner"})
    project = client.post("/api/v1/projects", json={"name": "P3 async"}).json()["item"]["project_id"]
    run_response = client.post("/api/v1/runs", json={"project_id": project, "robot": {"robot_id": "unitree_g1_29dof"}, "motion": {"train_motion_sha256": "a" * 64}, "reward_config_sha256": "b" * 64, "training_config_sha256": "c" * 64})
    run_id = run_response.json()["item"]["run_id"]
    response = client.post(f"/api/v1/runs/{run_id}/train", headers={"X-User-Id": "p3-async-owner", "X-Worker-Id": "p3-worker", "X-Execution-Mode": "async"}, json={"motion_asset_version_id": "motion-1"})
    assert response.status_code == 202, response.text
    assert response.json()["submission"]["status"] == "QUEUED"
    run = client.get(f"/api/v1/runs/{run_id}").json()["item"]
    assert run["status"] == "TRAINING_PREPARING"
