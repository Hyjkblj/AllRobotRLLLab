from fastapi.testclient import TestClient

from backend.app.main import app


def test_robot_contract_endpoint() -> None:
    response = TestClient(app).get("/api/v1/robots")
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["robot_id"] == "unitree_g1_29dof"
    assert body["items"][0]["dof"] == 29


def test_training_schema_endpoint() -> None:
    response = TestClient(app).get("/api/v1/training-config/schema")
    assert response.status_code == 200
    assert response.json()["schema"]["properties"]["ppo"]

