from pathlib import Path

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.config_builder import build_imitation_config
from backend.app.application.training_validator import resume_is_compatible, validate_training_config
from backend.app.domain.contracts import ArrayField, TrainMotionNPZ


def _config():
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    motion = TrainMotionNPZ(
        robot_id=adapter.name,
        fps=30,
        frame_count=30,
        joint_names=adapter.get_spec().joint_names,
        body_names=adapter.get_spec().body_names,
        arrays={"joint_pos": ArrayField(path="joint_pos", shape=[30, 29], dtype="float32"), "joint_vel": ArrayField(path="joint_vel", shape=[30, 29], dtype="float32"), "body_pos_w": ArrayField(path="body_pos", shape=[30, 6, 3], dtype="float32"), "body_quat_w": ArrayField(path="body_quat", shape=[30, 6, 4], dtype="float32", convention="wxyz"), "body_lin_vel_w": ArrayField(path="body_lin", shape=[30, 6, 3], dtype="float32"), "body_ang_vel_w": ArrayField(path="body_ang", shape=[30, 6, 3], dtype="float32")},
        coord_frame="world_z_up", quat_convention="wxyz", source_motion_hash="a" * 64, compiler_version="test",
    )
    return adapter, build_imitation_config(motion, adapter.get_spec()).config


def test_training_config_is_checked_against_adapter() -> None:
    adapter, config = _config()
    assert validate_training_config(config, adapter.get_spec()).valid
    invalid = config.__class__.model_validate({**config.model_dump(mode="json"), "action": {"mode": "joint_position_delta", "scale": 0.5, "clip": 1.0}})
    result = validate_training_config(invalid, adapter.get_spec())
    assert not result.valid
    assert any(issue.code == "ACTION_SCALE_EXCEEDS_ADAPTER" for issue in result.issues)


def test_resume_compatibility_rejects_structural_motion_changes() -> None:
    adapter, config = _config()
    changed_motion = config.model_copy(update={"motion_asset_version_id": "new-motion"})
    assert not resume_is_compatible(config, changed_motion)
    changed_learning_rate = config.__class__.model_validate({**config.model_dump(mode="json"), "ppo": {**config.ppo.model_dump(mode="json"), "learning_rate": 0.0005}})
    assert resume_is_compatible(config, changed_learning_rate)
