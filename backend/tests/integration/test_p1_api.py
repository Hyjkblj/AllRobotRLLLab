from fastapi.testclient import TestClient

from backend.app.application.reward_catalog import default_reward_config
from backend.app.main import app


G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def test_motion_edit_create_and_compile_api() -> None:
    client = TestClient(app)
    created = client.post("/api/v1/motion-edits", json={"source_motion_version_id": "api-motion-1", "robot_id": "unitree_g1_29dof", "global_transform": {"translation": [0.0, 0.0, 0.01], "yaw_offset": 0.0, "time_scale": 1.0}, "joint_offsets": [], "ik_targets": [], "keyframes": [], "filters": {"smooth": True, "max_velocity_check": True}})
    assert created.status_code == 200
    version_id = created.json()["item"]["version_id"]
    compiled = client.post(f"/api/v1/motion-edits/{version_id}/compile", json={"fps": 10, "joint_pos": [[0.0] * 29 for _ in range(3)], "root_pos": [[0.0, 0.0, 0.8] for _ in range(3)], "root_rot": [[0.0, 0.0, 0.0, 1.0] for _ in range(3)], "joint_names": G1_JOINT_NAMES})
    assert compiled.status_code == 200
    assert compiled.json()["quality"]["status"] == "PASS"
    assert compiled.json()["motion"]["frame_count"] == 3


def test_reward_config_version_api() -> None:
    client = TestClient(app)
    created = client.post("/api/v1/reward-configs", json=default_reward_config().model_dump(mode="json"))
    assert created.status_code == 200
    version_id = created.json()["item"]["version_id"]
    fetched = client.get(f"/api/v1/reward-configs/{version_id}")
    assert fetched.status_code == 200
    assert fetched.json()["item"]["config"]["base_template"] == "g1_mimic_v1"

