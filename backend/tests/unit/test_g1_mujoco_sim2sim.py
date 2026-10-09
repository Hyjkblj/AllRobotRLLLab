from pathlib import Path

import numpy as np
import pytest

from apps.mujoco_sim2sim.g1 import (
    BASE_OBSERVATION_DIM,
    BODY_NAMES,
    JOINT_NAMES,
    Policy,
    ROBOT_ID,
    TRAIN_MOTION_FORMAT,
    _control_gains,
    _quat_error_deg,
    evaluate,
)


def test_g1_observation_and_control_contract() -> None:
    assert len(JOINT_NAMES) == 29
    assert BASE_OBSERVATION_DIM == 122
    kp, kd = _control_gains()
    assert kp.shape == kd.shape == (29,)
    assert np.all(kp > 0.0)
    assert np.all(kd > 0.0)


def test_quaternion_error_is_sign_invariant() -> None:
    quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
    assert _quat_error_deg(quaternion, quaternion) == pytest.approx(0.0)
    assert _quat_error_deg(quaternion, -quaternion) == pytest.approx(0.0)


def test_torchscript_policy_uses_explicit_observation_dimension(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    observation_dim = BASE_OBSERVATION_DIM * 3 + 10
    model = torch.nn.Sequential(torch.nn.Linear(observation_dim, len(JOINT_NAMES)), torch.nn.Tanh()).eval()
    path = tmp_path / "policy.pt"
    torch.jit.trace(model, torch.zeros((1, observation_dim))).save(str(path))
    policy = Policy(path, observation_dim=observation_dim)
    result = policy.infer(np.zeros(observation_dim, dtype=np.float32))
    assert result.shape == (len(JOINT_NAMES),)


def test_real_unitree_mujoco_model_executes_project_policy_loop(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    torch = pytest.importorskip("torch")
    repository = Path(__file__).resolve().parents[3]
    model_path = repository / "third_party" / "unitree_mujoco-main" / "unitree_robots" / "g1" / "scene_29dof.xml"
    if not model_path.is_file():
        pytest.skip("Unitree MuJoCo source checkout is not present")

    observation_dim = BASE_OBSERVATION_DIM * 3 + 10
    policy_model = torch.nn.Linear(observation_dim, len(JOINT_NAMES)).eval()
    with torch.no_grad():
        policy_model.weight.zero_()
        policy_model.bias.zero_()
    policy_path = tmp_path / "policy.pt"
    torch.jit.trace(policy_model, torch.zeros((1, observation_dim))).save(str(policy_path))

    frames = 3
    body_count = len(BODY_NAMES)
    body_pos = np.zeros((frames, body_count, 3), dtype=np.float32)
    body_pos[:, 0, 2] = 0.793
    body_quat = np.zeros((frames, body_count, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    motion_path = tmp_path / "motion.npz"
    np.savez_compressed(
        motion_path,
        format_version=np.asarray(TRAIN_MOTION_FORMAT),
        robot_id=np.asarray(ROBOT_ID),
        fps=np.asarray(50.0),
        joint_names=np.asarray(JOINT_NAMES),
        body_names=np.asarray(BODY_NAMES),
        coord_frame=np.asarray("world_z_up"),
        quat_convention=np.asarray("wxyz"),
        source_motion_hash=np.asarray("0" * 64),
        compiler_version=np.asarray("test.v1"),
        joint_pos=np.zeros((frames, len(JOINT_NAMES)), dtype=np.float32),
        joint_vel=np.zeros((frames, len(JOINT_NAMES)), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, body_count, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, body_count, 3), dtype=np.float32),
    )

    output = tmp_path / "output"
    metrics = evaluate(
        seed=7,
        policy_path=policy_path,
        motion_path=motion_path,
        model_path=model_path,
        output_dir=output,
        max_steps=2,
        observation_dim=observation_dim,
    )
    assert set(metrics) == {
        "survival_rate", "joint_rmse_rad", "root_position_rmse_m",
        "orientation_error_deg", "saturation_ratio", "foot_slip_mps",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert (output / "metrics.json").is_file()
    assert (output / "trace.csv").is_file()
    assert (output / "seed_report.json").is_file()
