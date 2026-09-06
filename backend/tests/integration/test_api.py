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
    assert response.json()["tasks"][0]["task_id"] == "g1_mimic"


def test_task_endpoint_can_filter_by_robot() -> None:
    response = TestClient(app).get("/api/v1/tasks", params={"robot_id": "unitree_g1_29dof"})
    assert response.status_code == 200
    assert response.json()["items"][0]["robot_id"] == "unitree_g1_29dof"


def test_reward_catalog_and_validation_accept_explicit_scope() -> None:
    client = TestClient(app)
    listed = client.get("/api/v1/reward-templates", params={"robot_id": "unitree_g1_29dof", "task_id": "g1_mimic"})
    assert listed.status_code == 200
    assert listed.json()["items"]
    from backend.app.application.reward_catalog import default_reward_config

    payload = default_reward_config().model_dump(mode="json")
    response = client.post("/api/v1/reward-configs/validate", params={"robot_id": "unitree_g1_29dof", "task_id": "g1_mimic"}, json=payload)
    assert response.status_code == 200
    assert response.json()["result"]["valid"] is True


def test_reward_scope_rejects_task_for_another_robot() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/reward-templates", params={"robot_id": "unknown", "task_id": "g1_mimic"})
    assert response.status_code == 404
