from pathlib import Path

import numpy as np
import pytest

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.motion_editor import MotionArrays, MotionEditVersionStore, MotionEditor
from backend.app.domain.contracts import MotionEditConfig


def _arrays(adapter: UnitreeG1Adapter, frames: int = 5) -> MotionArrays:
    return MotionArrays(
        fps=10,
        joint_pos=np.zeros((frames, adapter.get_spec().dof), dtype=np.float64),
        root_pos=np.tile(np.asarray([[0.0, 0.0, 0.8]]), (frames, 1)),
        root_rot=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]]), (frames, 1)),
        joint_names=tuple(adapter.get_spec().joint_names),
    )


def test_motion_edit_applies_transform_offset_keyframe_and_time_scale() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    joint = adapter.get_spec().joint_names[0]
    config = MotionEditConfig(
        source_motion_version_id="motion-1",
        robot_id=adapter.name,
        global_transform={"translation": [0.1, 0.0, 0.03], "yaw_offset": 0.1, "time_scale": 2.0},
        joint_offsets=[{"joint_name": joint, "frame_start": 1, "frame_end": 3, "position_offset": 0.1}],
        keyframes=[{"frame": 0, "qpos": [0.2] + [0.0] * 28}, {"frame": 4, "qpos": [0.4] + [0.0] * 28}],
    )
    result = MotionEditor(adapter.get_spec()).apply(_arrays(adapter), config)
    assert result.validation.valid
    assert result.quality.status == "PASS"
    assert result.arrays is not None
    assert len(result.arrays.joint_pos) == 9
    assert np.isclose(result.arrays.root_pos[0, 2], 0.83)
    assert np.isclose(result.arrays.joint_pos[0, 0], 0.2)


def test_motion_edit_blocks_hard_joint_limit_violation() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    config = MotionEditConfig(source_motion_version_id="motion-1", robot_id=adapter.name, joint_offsets=[{"joint_name": adapter.get_spec().joint_names[0], "frame_start": 0, "frame_end": 4, "position_offset": 10.0}])
    result = MotionEditor(adapter.get_spec()).apply(_arrays(adapter), config)
    assert not result.validation.valid
    assert result.arrays is not None
    assert any(issue.code == "JOINT_LIMIT_VIOLATION" for issue in result.quality.issues)


def test_motion_edit_requires_server_side_ik_solver() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    config = MotionEditConfig(source_motion_version_id="motion-1", robot_id=adapter.name, ik_targets=[{"body_name": "left_rubber_hand", "frame_start": 0, "frame_end": 2, "target_offset": [0.01, 0.0, 0.0]}])
    result = MotionEditor(adapter.get_spec()).apply(_arrays(adapter), config)
    assert not result.validation.valid
    assert any(issue.code == "IK_SOLVER_UNAVAILABLE" for issue in result.quality.issues)


def test_g1_mujoco_ik_solver_accepts_zero_offset_target() -> None:
    pytest.importorskip("mujoco")
    from adapters.unitree_g1_29dof.ik_solver import G1MuJoCoIKSolver

    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    solved = G1MuJoCoIKSolver(repository_root=Path(__file__).resolve().parents[3]).apply(_arrays(adapter, frames=1), [{"body_name": "left_rubber_hand", "frame_start": 0, "frame_end": 0, "target_offset": [0.0, 0.0, 0.0]}])
    assert solved.joint_pos.shape == (1, 29)


def test_motion_edit_versions_are_immutable_and_chained() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    store = MotionEditVersionStore()
    first = store.create(MotionEditConfig(source_motion_version_id="motion-1", robot_id=adapter.name))
    second = store.create(MotionEditConfig(source_motion_version_id="motion-1", robot_id=adapter.name, global_transform={"translation": [0.0, 0.0, 0.01]}), parent_version_id=first.version_id)
    assert first.version == 1
    assert second.version == 2
    assert second.parent_version_id == first.version_id
    assert len(store.list_for_source("motion-1")) == 2
    restored = store.restore(second.version_id, first.version_id)
    assert restored.parent_version_id == second.version_id
    assert restored.config_sha256 == first.config_sha256
