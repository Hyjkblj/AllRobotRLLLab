from pathlib import Path

import numpy as np
import pytest

from backend.app.adapters.motion import MotionDetectionError, MotionSourceRegistry


def test_npz_g1_trajectory_is_content_detected(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(path, qpos=np.zeros((20, 29), dtype=np.float32), fps=np.asarray(30, dtype=np.int32))
    descriptor = MotionSourceRegistry(default_dof=29, default_robot_id="unitree_g1_29dof").detect(path)
    assert descriptor.detected_type == "g1_joint_trajectory"
    assert descriptor.fields["qpos"].shape == [20, 29]


def test_npz_wrong_dof_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, joint_pos=np.zeros((20, 28), dtype=np.float32))
    with pytest.raises(MotionDetectionError) as error:
        MotionSourceRegistry(default_dof=29, default_robot_id="unitree_g1_29dof").detect(path)
    assert error.value.code == "SCHEMA_INVALID"


def test_pickle_is_rejected_without_trust_flag(tmp_path: Path) -> None:
    path = tmp_path / "motion.pkl"
    path.write_bytes(b"not trusted")
    with pytest.raises(MotionDetectionError) as error:
        MotionSourceRegistry().detect(path)
    assert error.value.code == "UNTRUSTED_PICKLE"


def test_detector_accepts_unscoped_non_g1_dof_when_registry_is_generic(tmp_path: Path) -> None:
    path = tmp_path / "fixture.npz"
    np.savez(path, joint_pos=np.zeros((20, 2), dtype=np.float32))
    descriptor = MotionSourceRegistry(default_dof=None, default_robot_id="fixture_biped_2dof").detect(path)
    assert descriptor.detected_type == "joint_trajectory"
    assert descriptor.fields["joint_pos"].shape == [20, 2]


def test_unscoped_detector_is_vendor_neutral(tmp_path: Path) -> None:
    path = tmp_path / "generic.npz"
    np.savez(path, joint_pos=np.zeros((20, 3), dtype=np.float32))
    descriptor = MotionSourceRegistry().detect(path)
    assert descriptor.detected_type == "joint_trajectory"
    assert descriptor.source_skeleton is None
