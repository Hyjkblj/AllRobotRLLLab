"""Generate a reviewable G1 imitation training configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from backend.app.domain.contracts import RobotSpec, TrainMotionNPZ, TrainingConfig
from .reward_catalog import default_reward_config


@dataclass(frozen=True)
class GeneratedImitationConfig:
    config: TrainingConfig
    derived: dict[str, Any]
    defaults: dict[str, Any]
    user_overrides: dict[str, Any]
    builder_version: str = "imitation-config-builder.v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "generated_imitation_config.v1",
            "source_motion_sha256": "pending",
            "robot_spec_sha256": "pending",
            "derived": self.derived,
            "defaults": self.defaults,
            "user_overrides": self.user_overrides,
            "builder_version": self.builder_version,
            "config": self.config.model_dump(mode="json"),
            "reward_config": default_reward_config().model_dump(mode="json"),
        }


def build_imitation_config(
    motion: TrainMotionNPZ,
    robot: RobotSpec,
    *,
    user_overrides: dict[str, Any] | None = None,
) -> GeneratedImitationConfig:
    overrides = user_overrides or {}
    motion_duration = motion.frame_count / motion.fps
    policy_dt = robot.actuation.policy_dt
    horizon = max(1, min(10000, math.ceil(motion_duration / policy_dt)))
    # Position/velocity/gravity/action/reference/history are explicit so the
    # resulting dimension can be checked by the training adapter before launch.
    observation_dim = robot.dof * 2 + 3 + 3 + robot.dof + robot.dof
    observation_dim *= 3
    observation_dim += 6 + 4
    config_data: dict[str, Any] = {
        "task_id": "g1_mimic",
        "scene_id": "g1_flat",
        "motion_asset_version_id": "pending",
        "observation": {"history_length": 3, "include_root_velocity": True, "include_projected_gravity": True, "include_reference": True, "clip_value": 100.0},
        "action": {"mode": "joint_position_delta", "scale": robot.actuation.action_scale, "clip": 1.0},
        "control": {"decimation": max(1, round(robot.actuation.policy_dt / robot.actuation.control_dt)), "kp_profile": robot.actuation.kp_profile, "kd_profile": robot.actuation.kd_profile},
        "ppo": {"algorithm": "rsl_rl_ppo", "seed": 1234, "num_envs": 4096, "max_iterations": 5000, "rollout_length": 24, "learning_rate": 0.001, "schedule": "adaptive", "gamma": 0.99, "lam": 0.95, "clip_param": 0.2, "entropy_coef": 0.01, "value_loss_coef": 1.0, "max_grad_norm": 1.0, "hidden_dims": [512, 256, 128]},
        "domain_randomization": {"enabled": True, "mass_scale": [0.95, 1.05], "friction": [0.7, 1.3], "motor_strength": [0.95, 1.05]},
        "resources": {"gpu_count": 1, "gpu_memory_gb": 8, "cpu_cores": 8, "shared_memory_gb": 8, "exclusive_gpu": False},
    }
    for key, value in overrides.items():
        if key not in config_data:
            continue
        if isinstance(value, dict) and isinstance(config_data[key], dict):
            config_data[key] = {**config_data[key], **value}
        else:
            config_data[key] = value
    config = TrainingConfig.model_validate(config_data)
    return GeneratedImitationConfig(
        config=config,
        derived={"policy_dt": policy_dt, "episode_horizon": horizon, "observation_dim": observation_dim, "action_dim": robot.dof, "active_reward_terms": ["tracking.joint_pos", "tracking.root_pose", "regularization.action_rate"]},
        defaults=config_data,
        user_overrides=overrides,
    )


__all__ = ["GeneratedImitationConfig", "build_imitation_config"]

