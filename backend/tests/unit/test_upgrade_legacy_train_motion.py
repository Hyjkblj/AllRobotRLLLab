from pathlib import Path

import numpy as np
import pytest

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.train_motion_validator import TrainMotionFileError, validate_train_motion_file
from backend.app.config.settings import settings
from scripts.upgrade_legacy_train_motion import sha256_file, upgrade_legacy_train_motion


def _legacy_archive(path: Path) -> None:
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
    frames = 4
    bodies = len(robot.body_names)
    np.savez_compressed(
        path,
        joint_pos=np.zeros((frames, robot.dof), dtype=np.float32),
        joint_vel=np.zeros((frames, robot.dof), dtype=np.float32),
        body_pos_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        body_quat_w=np.broadcast_to(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (frames, bodies, 4),
        ).copy(),
        body_lin_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        fps=np.asarray(50.0, dtype=np.float32),
    )


def test_upgrade_legacy_train_motion_adds_valid_identity(tmp_path: Path) -> None:
    source = tmp_path / "legacy.npz"
    output = tmp_path / "train_motion.npz"
    _legacy_archive(source)
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()

    result = upgrade_legacy_train_motion(
        source,
        output,
        robot=robot,
        assume_robot_spec_order=True,
    )

    validate_train_motion_file(output, robot, expected_sha256=str(result["sha256"]))
    assert result["source_sha256"] == sha256_file(source)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["robot_id"].item() == robot.robot_id
        assert archive["source_motion_hash"].item() == sha256_file(source)
        assert archive["joint_pos"].dtype == np.dtype(np.float32)


def test_upgrade_legacy_train_motion_preserves_source_and_array_dtype(tmp_path: Path) -> None:
    source = tmp_path / "legacy.npz"
    output = tmp_path / "train_motion.npz"
    _legacy_archive(source)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["joint_pos"] = payload["joint_pos"].astype(np.float64)
    np.savez_compressed(source, **payload)
    source_before = source.read_bytes()
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()

    upgrade_legacy_train_motion(
        source,
        output,
        robot=robot,
        assume_robot_spec_order=True,
    )

    assert source.read_bytes() == source_before
    with np.load(output, allow_pickle=False) as archive:
        assert archive["joint_pos"].dtype == np.dtype(np.float64)


def test_upgrade_legacy_train_motion_requires_explicit_order_acknowledgement(tmp_path: Path) -> None:
    source = tmp_path / "legacy.npz"
    _legacy_archive(source)
    robot = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()

    with pytest.raises(TrainMotionFileError, match="assume-robot-spec-order"):
        upgrade_legacy_train_motion(
            source,
            tmp_path / "train_motion.npz",
            robot=robot,
            assume_robot_spec_order=False,
        )
