from pathlib import Path

import numpy as np

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.reward_catalog import default_reward_config, validate_reward_config
from backend.app.application.config_builder import build_imitation_config
from backend.app.domain.contracts import ArrayField, RunManifest, TrainMotionNPZ
from backend.app.domain.state_machine import InvalidTransition, RunStatus, transition


def test_g1_spec_and_xml_contract_are_consistent() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    spec = adapter.get_spec()
    assert spec.dof == 29
    assert adapter.self_check().valid


def test_manifest_freeze_uses_canonical_hash() -> None:
    manifest = RunManifest(
        project_id="project-1",
        run_id="run-1",
        attempt_id="attempt-1",
        robot={"robot_id": "unitree_g1_29dof"},
        motion={"train_motion_sha256": "a" * 64},
        reward_config_sha256="b" * 64,
        training_config_sha256="c" * 64,
    )
    frozen = manifest.freeze()
    assert frozen.manifest_sha256 == frozen.computed_hash()
    assert frozen.canonical_bytes() == frozen.model_copy(update={"manifest_sha256": None}).canonical_bytes()


def test_safety_terminations_cannot_be_removed() -> None:
    config = default_reward_config().model_copy(update={"terminations": ["timeout"]})
    result = validate_reward_config(config)
    assert not result.valid
    assert any(issue.code == "SAFETY_TERMINATION_REQUIRED" for issue in result.issues)


def test_state_machine_rejects_skipping_training() -> None:
    assert transition(RunStatus.CREATED, RunStatus.UPLOADING) == RunStatus.UPLOADING
    try:
        transition(RunStatus.CREATED, RunStatus.TRAINING)
    except InvalidTransition:
        pass
    else:
        raise AssertionError("state machine accepted an invalid transition")


def test_imitation_config_derives_g1_dimensions() -> None:
    adapter = UnitreeG1Adapter(repository_root=Path(__file__).resolve().parents[3])
    motion = TrainMotionNPZ(
        robot_id=adapter.name,
        fps=30,
        frame_count=60,
        joint_names=adapter.get_spec().joint_names,
        body_names=adapter.get_spec().body_names,
        arrays={
            "joint_pos": ArrayField(path="joint_pos.npy", shape=[60, 29], dtype="float32"),
            "joint_vel": ArrayField(path="joint_vel.npy", shape=[60, 29], dtype="float32"),
            "body_pos_w": ArrayField(path="body_pos_w.npy", shape=[60, 6, 3], dtype="float32"),
            "body_quat_w": ArrayField(path="body_quat_w.npy", shape=[60, 6, 4], dtype="float32", convention="wxyz"),
            "body_lin_vel_w": ArrayField(path="body_lin_vel_w.npy", shape=[60, 6, 3], dtype="float32"),
            "body_ang_vel_w": ArrayField(path="body_ang_vel_w.npy", shape=[60, 6, 3], dtype="float32"),
        },
        coord_frame="world_z_up",
        quat_convention="wxyz",
        source_motion_hash="a" * 64,
        compiler_version="test",
    )
    generated = build_imitation_config(motion, adapter.get_spec())
    assert generated.derived["action_dim"] == 29
    assert generated.derived["episode_horizon"] == 100
