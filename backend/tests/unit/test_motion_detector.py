from pathlib import Path

import numpy as np
import pytest

from backend.app.adapters.motion import MotionDetectionError, MotionSourceRegistry


def test_npz_g1_trajectory_is_content_detected(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    np.savez(path, qpos=np.zeros((20, 29), dtype=np.float32), fps=np.asarray(30, dtype=np.int32))
    descriptor = MotionSourceRegistry().detect(path)
    assert descriptor.detected_type == "g1_joint_trajectory"
    assert descriptor.fields["qpos"].shape == [20, 29]


def test_npz_wrong_dof_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, joint_pos=np.zeros((20, 28), dtype=np.float32))
    with pytest.raises(MotionDetectionError) as error:
        MotionSourceRegistry().detect(path)
    assert error.value.code == "SCHEMA_INVALID"


def test_pickle_is_rejected_without_trust_flag(tmp_path: Path) -> None:
    path = tmp_path / "motion.pkl"
    path.write_bytes(b"not trusted")
    with pytest.raises(MotionDetectionError) as error:
        MotionSourceRegistry().detect(path)
    assert error.value.code == "UNTRUSTED_PICKLE"
