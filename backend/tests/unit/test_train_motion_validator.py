import hashlib
from pathlib import Path

import numpy as np
import pytest

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.train_motion_validator import TrainMotionFileError, validate_train_motion_file
from backend.app.config.settings import settings


def _write_motion(path: Path, **updates) -> None:
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
    frames = 3
    bodies = len(robot.body_names)
    payload = {
        "joint_pos": np.zeros((frames, robot.dof), dtype=np.float32),
        "joint_vel": np.zeros((frames, robot.dof), dtype=np.float32),
        "body_pos_w": np.zeros((frames, bodies, 3), dtype=np.float32),
        "body_quat_w": np.broadcast_to(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, bodies, 4)).copy(),
        "body_lin_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
        "format_version": np.asarray("train_motion_npz.v1"),
        "robot_id": np.asarray(robot.robot_id),
        "fps": np.asarray(30.0, dtype=np.float32),
        "joint_names": np.asarray(robot.joint_names),
        "body_names": np.asarray(robot.body_names),
        "coord_frame": np.asarray("world_z_up"),
        "quat_convention": np.asarray("wxyz"),
        "source_motion_hash": np.asarray("a" * 64),
        "compiler_version": np.asarray("test-compiler.v1"),
    }
    payload.update(updates)
    np.savez_compressed(path, **payload)


def test_train_motion_file_matches_robot_contract(tmp_path: Path) -> None:
    path = tmp_path / "train_motion.npz"
    _write_motion(path)
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
    validate_train_motion_file(path, robot, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"robot_id": np.asarray("unitree_h1_19dof")}, "robot_id"),
        ({"joint_names": np.asarray(["wrong"] * 29)}, "joint_names"),
        ({"quat_convention": np.asarray("xyzw")}, "quat_convention"),
        ({"joint_pos": np.full((3, 29), np.nan, dtype=np.float32)}, "non-finite"),
        ({"body_quat_w": np.zeros((3, 6, 4), dtype=np.float32)}, "not normalized"),
    ],
)
def test_train_motion_file_rejects_contract_mismatch(tmp_path: Path, updates, message: str) -> None:
    path = tmp_path / "invalid.npz"
    _write_motion(path, **updates)
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
    with pytest.raises(TrainMotionFileError, match=message):
        validate_train_motion_file(path, robot)


def test_train_motion_file_rejects_missing_identity_metadata(tmp_path: Path) -> None:
    path = tmp_path / "source_only.npz"
    np.savez(path, joint_pos=np.zeros((20, 29), dtype=np.float32), fps=np.asarray(30.0))
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
    with pytest.raises(TrainMotionFileError, match="missing fields"):
        validate_train_motion_file(path, robot)


def test_train_motion_file_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "train_motion.npz"
    _write_motion(path)
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
    with pytest.raises(TrainMotionFileError, match="Run Manifest hash"):
        validate_train_motion_file(path, robot, expected_sha256="b" * 64)
