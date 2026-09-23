"""Platform-owned Isaac Lab/RSL-RL implementation of the G1 mimic task.

This module is imported only after :class:`isaaclab.app.AppLauncher` starts.
It deliberately depends on Isaac Lab and RSL-RL, but never imports
``unitree_rl_lab``.  The task consumes the platform TrainMotionNPZ contract
and a G1 URDF supplied by the robot runtime registration.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from rsl_rl.runners import OnPolicyRunner


TASK_ID = "g1_mimic"
JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
BASE_OBSERVATION_DIM = len(JOINT_NAMES) * 4 + 6
G1_CONTROL_DT = 0.02
CHECKPOINT_CONTEXT_KEY = "allrobotrl_context"
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQUENCY = 10.0 * 2.0 * np.pi
DAMPING_RATIO = 2.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQUENCY**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQUENCY**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQUENCY**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQUENCY**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQUENCY
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQUENCY
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQUENCY
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQUENCY


def _required_file(env_name: str) -> Path:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required by the native G1 Isaac task")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{env_name} does not point to a file: {path}")
    return path


def _robot_cfg() -> ArticulationCfg:
    urdf = _required_file("G1_ISAAC_URDF_PATH")
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            fix_base=False,
            activate_contact_sensors=True,
            replace_cylinders_with_capsules=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_elbow_joint": 0.6,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_pitch_joint": 0.2,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                },
                stiffness={
                    ".*_hip_pitch_joint": STIFFNESS_7520_14,
                    ".*_hip_roll_joint": STIFFNESS_7520_22,
                    ".*_hip_yaw_joint": STIFFNESS_7520_14,
                    ".*_knee_joint": STIFFNESS_7520_22,
                },
                damping={
                    ".*_hip_pitch_joint": DAMPING_7520_14,
                    ".*_hip_roll_joint": DAMPING_7520_22,
                    ".*_hip_yaw_joint": DAMPING_7520_14,
                    ".*_knee_joint": DAMPING_7520_22,
                },
                armature={
                    ".*_hip_pitch_joint": ARMATURE_7520_14,
                    ".*_hip_roll_joint": ARMATURE_7520_22,
                    ".*_hip_yaw_joint": ARMATURE_7520_14,
                    ".*_knee_joint": ARMATURE_7520_22,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=37.0,
                stiffness=2.0 * STIFFNESS_5020,
                damping=2.0 * DAMPING_5020,
                armature=2.0 * ARMATURE_5020,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
                effort_limit_sim=50.0,
                velocity_limit_sim=37.0,
                stiffness=2.0 * STIFFNESS_5020,
                damping=2.0 * DAMPING_5020,
                armature=2.0 * ARMATURE_5020,
            ),
            "waist_yaw": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint"],
                effort_limit_sim=88.0,
                velocity_limit_sim=32.0,
                stiffness=STIFFNESS_7520_14,
                damping=DAMPING_7520_14,
                armature=ARMATURE_7520_14,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                    ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
                ],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 25.0, ".*_shoulder_roll_joint": 25.0,
                    ".*_shoulder_yaw_joint": 25.0, ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0, ".*_wrist_pitch_joint": 5.0, ".*_wrist_yaw_joint": 5.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 37.0, ".*_shoulder_roll_joint": 37.0,
                    ".*_shoulder_yaw_joint": 37.0, ".*_elbow_joint": 37.0,
                    ".*_wrist_roll_joint": 37.0, ".*_wrist_pitch_joint": 22.0, ".*_wrist_yaw_joint": 22.0,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": STIFFNESS_5020, ".*_shoulder_roll_joint": STIFFNESS_5020,
                    ".*_shoulder_yaw_joint": STIFFNESS_5020, ".*_elbow_joint": STIFFNESS_5020,
                    ".*_wrist_roll_joint": STIFFNESS_5020, ".*_wrist_pitch_joint": STIFFNESS_4010,
                    ".*_wrist_yaw_joint": STIFFNESS_4010,
                },
                damping={
                    ".*_shoulder_pitch_joint": DAMPING_5020, ".*_shoulder_roll_joint": DAMPING_5020,
                    ".*_shoulder_yaw_joint": DAMPING_5020, ".*_elbow_joint": DAMPING_5020,
                    ".*_wrist_roll_joint": DAMPING_5020, ".*_wrist_pitch_joint": DAMPING_4010,
                    ".*_wrist_yaw_joint": DAMPING_4010,
                },
                armature={
                    ".*_shoulder_pitch_joint": ARMATURE_5020, ".*_shoulder_roll_joint": ARMATURE_5020,
                    ".*_shoulder_yaw_joint": ARMATURE_5020, ".*_elbow_joint": ARMATURE_5020,
                    ".*_wrist_roll_joint": ARMATURE_5020, ".*_wrist_pitch_joint": ARMATURE_4010,
                    ".*_wrist_yaw_joint": ARMATURE_4010,
                },
            ),
        },
    )


def _resample_motion(arrays: dict[str, np.ndarray], *, source_fps: float, target_fps: float) -> dict[str, np.ndarray]:
    """Resample the immutable motion contract to the policy control rate."""

    frame_count = int(arrays["joint_pos"].shape[0])
    if frame_count < 2 or abs(source_fps - target_fps) < 1.0e-6:
        return arrays
    duration = (frame_count - 1) / source_fps
    target_count = max(2, int(round(duration * target_fps)) + 1)
    positions = np.linspace(0.0, frame_count - 1, target_count, dtype=np.float64)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, frame_count - 1)
    alpha = (positions - lower).astype(np.float32)
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        source = np.asarray(value, dtype=np.float32)
        blend = alpha.reshape((target_count,) + (1,) * (source.ndim - 1))
        if name != "body_quat_w":
            result[name] = source[lower] * (1.0 - blend) + source[upper] * blend
            continue
        first = source[lower]
        second = source[upper]
        dot = np.sum(first * second, axis=-1, keepdims=True)
        second = np.where(dot < 0.0, -second, second)
        dot = np.clip(np.abs(dot), 0.0, 1.0)
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        quat_blend = blend
        first_weight = np.where(
            sin_theta > 1.0e-6,
            np.sin((1.0 - quat_blend) * theta) / np.maximum(sin_theta, 1.0e-8),
            1.0 - quat_blend,
        )
        second_weight = np.where(
            sin_theta > 1.0e-6,
            np.sin(quat_blend * theta) / np.maximum(sin_theta, 1.0e-8),
            quat_blend,
        )
        quaternions = first_weight * first + second_weight * second
        result[name] = quaternions / np.maximum(np.linalg.norm(quaternions, axis=-1, keepdims=True), 1.0e-8)
    return result


@configclass
class G1DomainRandomizationCfg:
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )


@configclass
class G1MimicEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 30.0
    action_space = len(JOINT_NAMES)
    observation_space = BASE_OBSERVATION_DIM + 10
    state_space = 0
    action_scale = 0.25
    action_clip = 1.0
    observation_clip = 100.0
    history_length = 1
    termination_height_error = 0.30
    reward_weights: dict[str, float] = {}
    reward_sigmas: dict[str, float] = {}
    terminations: tuple[str, ...] = ("timeout", "bad_anchor_orientation", "fall", "joint_limit", "nan_inf")
    sim: SimulationCfg = SimulationCfg(dt=G1_CONTROL_DT, render_interval=decimation)
    robot_cfg: ArticulationCfg = _robot_cfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    motion_file: str = ""
    events: G1DomainRandomizationCfg | None = None


class G1MimicEnv(DirectRLEnv):
    cfg: G1MimicEnvCfg

    def __init__(self, cfg: G1MimicEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, resolved_joints = self.robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(resolved_joints) != JOINT_NAMES:
            raise RuntimeError("G1 URDF joint order does not satisfy the platform RobotSpec")
        self._body_ids, resolved_bodies = self.robot.find_bodies(list(BODY_NAMES), preserve_order=True)
        if tuple(resolved_bodies) != BODY_NAMES:
            raise RuntimeError("G1 URDF body set does not satisfy the platform RobotSpec")
        self._load_motion(Path(cfg.motion_file))
        self.actions = torch.zeros((self.num_envs, len(JOINT_NAMES)), dtype=torch.float32, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self._motion_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motion_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._observation_history = torch.zeros(
            (self.num_envs, cfg.history_length, BASE_OBSERVATION_DIM), dtype=torch.float32, device=self.device
        )
        self._observation_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._observation_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _load_motion(self, path: Path) -> None:
        if not path.is_file():
            raise RuntimeError(f"TrainMotionNPZ does not exist: {path}")
        with np.load(path, allow_pickle=False) as data:
            joint_names = tuple(str(value) for value in np.asarray(data["joint_names"]).tolist())
            body_names = tuple(str(value) for value in np.asarray(data["body_names"]).tolist())
            if joint_names != JOINT_NAMES or body_names != BODY_NAMES:
                raise RuntimeError("TrainMotionNPZ ordering does not match the native G1 task")
            source_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            raw_arrays = {
                name: np.asarray(data[name], dtype=np.float32)
                for name in (
                    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"
                )
            }
        target_fps = 1.0 / float(self.cfg.sim.dt * self.cfg.decimation)
        resampled = _resample_motion(raw_arrays, source_fps=source_fps, target_fps=target_fps)
        arrays = {
            name: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for name, value in resampled.items()
        }
        self.motion_source_fps = source_fps
        self.motion_fps = target_fps
        self.motion_joint_pos = arrays["joint_pos"]
        self.motion_joint_vel = arrays["joint_vel"]
        self.motion_body_pos = arrays["body_pos_w"]
        self.motion_body_quat = arrays["body_quat_w"]
        self.motion_body_lin_vel = arrays["body_lin_vel_w"]
        self.motion_body_ang_vel = arrays["body_ang_vel_w"]
        self.motion_frame_count = int(self.motion_joint_pos.shape[0])
        if self.motion_frame_count < 2:
            raise RuntimeError("native G1 training requires at least two motion frames")

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _reference_body_pos(self) -> torch.Tensor:
        value = self.motion_body_pos[self._motion_step].clone()
        value[..., :2] += self.scene.env_origins[:, None, :2]
        return value

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions = torch.clamp(actions, -self.cfg.action_clip, self.cfg.action_clip)
        target = self.motion_joint_pos[self._motion_step] + self.cfg.action_scale * self.actions
        limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]
        self._joint_target = torch.clamp(target, limits[..., 0], limits[..., 1])

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_target, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # DirectRLEnv may expose observations more than once per physics step
        # (for example through a wrapper probe). Advance reference/history only
        # once for each environment step so reads remain side-effect-idempotent.
        update = (~self._observation_valid) | (self._observation_step != self.episode_length_buf)
        self._motion_complete[update] = self._motion_step[update] >= self.motion_frame_count - 1
        self._motion_step[update] = torch.clamp(self._motion_step[update] + 1, max=self.motion_frame_count - 1)
        current = torch.cat(
            (
                self.robot.data.joint_pos[:, self._joint_ids],
                self.robot.data.joint_vel[:, self._joint_ids],
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.motion_joint_pos[self._motion_step],
                self.motion_joint_vel[self._motion_step],
            ),
            dim=-1,
        )
        if torch.any(update):
            history = self._observation_history[update]
            history = torch.roll(history, shifts=1, dims=1)
            history[:, 0] = current[update]
            self._observation_history[update] = history
            self._observation_step[update] = self.episode_length_buf[update]
            self._observation_valid[update] = True
        reference_root = torch.cat(
            (self.motion_body_pos[self._motion_step, 0], self.motion_body_quat[self._motion_step, 0]), dim=-1
        )
        phase = (self._motion_step.float() / float(self.motion_frame_count)).unsqueeze(-1)
        phase_features = torch.cat((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase), phase), dim=-1)
        obs = torch.cat((self._observation_history.flatten(1), reference_root, phase_features), dim=-1)
        obs = torch.clamp(obs, -self.cfg.observation_clip, self.cfg.observation_clip)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        joint_error = torch.mean(
            torch.square(self.robot.data.joint_pos[:, self._joint_ids] - self.motion_joint_pos[self._motion_step]),
            dim=-1,
        )
        joint_velocity_error = torch.mean(
            torch.square(self.robot.data.joint_vel[:, self._joint_ids] - self.motion_joint_vel[self._motion_step]),
            dim=-1,
        )
        body_error = torch.mean(
            torch.sum(
                torch.square(self.robot.data.body_pos_w[:, self._body_ids] - self._reference_body_pos()),
                dim=-1,
            ),
            dim=-1,
        )
        orientation_error = torch.mean(
            torch.square(
                quat_error_magnitude(
                    self.motion_body_quat[self._motion_step],
                    self.robot.data.body_quat_w[:, self._body_ids],
                )
            ),
            dim=-1,
        )
        root_position_error = torch.sum(
            torch.square(self.robot.data.root_pos_w - self._reference_body_pos()[:, 0]), dim=-1
        )
        root_orientation_error = torch.square(
            quat_error_magnitude(self.motion_body_quat[self._motion_step, 0], self.robot.data.root_quat_w)
        )
        action_rate = torch.mean(torch.square(self.actions - self.previous_actions), dim=-1)
        torque = torch.mean(torch.square(self.robot.data.applied_torque[:, self._joint_ids]), dim=-1)
        foot_ids = [BODY_NAMES.index("left_ankle_roll_link"), BODY_NAMES.index("right_ankle_roll_link")]
        foot_velocity = self.robot.data.body_lin_vel_w[:, [self._body_ids[index] for index in foot_ids]]
        foot_height = self.robot.data.body_pos_w[:, [self._body_ids[index] for index in foot_ids], 2]
        foot_slip = torch.mean(torch.sum(torch.square(foot_velocity[..., :2]), dim=-1) * (foot_height < 0.08), dim=-1)
        expected_contact = torch.mean((foot_height < 0.08).float(), dim=-1)
        weights = self.cfg.reward_weights
        sigmas = self.cfg.reward_sigmas
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        reward += weights.get("tracking.joint_pos", 0.0) * torch.exp(
            -joint_error / sigmas.get("tracking.joint_pos", 0.25) ** 2
        )
        reward += weights.get("tracking.joint_vel", 0.0) * torch.exp(
            -joint_velocity_error / sigmas.get("tracking.joint_vel", 0.5) ** 2
        )
        reward += weights.get("tracking.root_pose", 0.0) * torch.exp(
            -(root_position_error + root_orientation_error) / sigmas.get("tracking.root_pose", 0.2) ** 2
        )
        reward += weights.get("tracking.body_pose", 0.0) * torch.exp(
            -(body_error + orientation_error) / sigmas.get("tracking.body_pose", 0.2) ** 2
        )
        reward += weights.get("regularization.action_rate", 0.0) * action_rate
        reward += weights.get("regularization.torque", 0.0) * torque
        reward += weights.get("stability.contact", 0.0) * expected_contact
        reward += weights.get("stability.foot_slip", 0.0) * foot_slip
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        reference_height = self.motion_body_pos[self._motion_step, 0, 2]
        enabled = set(self.cfg.terminations)
        fallen = (
            torch.abs(self.robot.data.root_pos_w[:, 2] - reference_height) > self.cfg.termination_height_error
            if "fall" in enabled
            else torch.zeros_like(time_out)
        )
        bad_orientation = (
            quat_error_magnitude(self.motion_body_quat[self._motion_step, 0], self.robot.data.root_quat_w) > 0.8
            if "bad_anchor_orientation" in enabled
            else torch.zeros_like(time_out)
        )
        non_finite = (
            ~torch.isfinite(self.robot.data.joint_pos[:, self._joint_ids]).all(dim=-1)
            if "nan_inf" in enabled
            else torch.zeros_like(time_out)
        )
        limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]
        joint_limit = (
            ((self.robot.data.joint_pos[:, self._joint_ids] < limits[..., 0]) | (self.robot.data.joint_pos[:, self._joint_ids] > limits[..., 1])).any(dim=-1)
            if "joint_limit" in enabled
            else torch.zeros_like(time_out)
        )
        timeout = time_out if "timeout" in enabled else torch.zeros_like(time_out)
        return fallen | bad_orientation | non_finite | joint_limit | self._motion_complete, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        frames = torch.randint(0, self.motion_frame_count - 1, (len(env_ids),), device=self.device)
        self._motion_step[env_ids] = frames
        self._motion_complete[env_ids] = False
        joint_pos = self.motion_joint_pos[frames]
        joint_vel = self.motion_joint_vel[frames]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.motion_body_pos[frames, 0]
        root_state[:, :2] += self.scene.env_origins[env_ids, :2]
        root_state[:, 3:7] = self.motion_body_quat[frames, 0]
        root_state[:, 7:10] = self.motion_body_lin_vel[frames, 0]
        root_state[:, 10:13] = self.motion_body_ang_vel[frames, 0]
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_ids, env_ids)
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self._observation_history[env_ids] = 0.0
        self._observation_valid[env_ids] = False
        self._observation_step[env_ids] = -1


@configclass
class G1MimicRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "allrobotrl_g1_mimic"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _training_input(manifest: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    manifest_payload = _read_json(manifest)
    config_raw = os.getenv("ALLROBOTRL_CONFIG", "").strip()
    if not config_raw:
        raise RuntimeError("ALLROBOTRL_CONFIG is required by the native G1 task")
    envelope = _read_json(Path(config_raw).expanduser().resolve())
    motion = Path(str(envelope.get("motion_path", os.getenv("ALLROBOTRL_MOTION", "")))).expanduser().resolve()
    config = envelope.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("native G1 training config payload is missing")
    if envelope.get("task_id") != TASK_ID or config.get("task_id") != TASK_ID:
        raise RuntimeError("native G1 training envelope task_id does not match g1_mimic")
    if not motion.is_file():
        raise RuntimeError(f"native G1 TrainMotionNPZ does not exist: {motion}")
    manifest_reward = str(manifest_payload.get("reward_config_sha256", ""))
    config_reward = str(config.get("reward_config_sha256", ""))
    if manifest_reward and config_reward != manifest_reward:
        raise RuntimeError("native G1 reward config identity does not match the frozen Run Manifest")
    return motion, config, manifest_payload


def _configs(*, motion: Path, config: dict[str, Any], device: str, num_envs: int | None = None):
    ppo = config.get("ppo") if isinstance(config.get("ppo"), dict) else {}
    env_cfg = G1MimicEnvCfg()
    env_cfg.motion_file = str(motion)
    env_cfg.scene.num_envs = int(num_envs if num_envs is not None else ppo.get("num_envs", 4096))
    env_cfg.seed = int(ppo.get("seed", 1234))
    env_cfg.sim.device = device
    observation = config.get("observation") if isinstance(config.get("observation"), dict) else {}
    required_observation_flags = ("include_root_velocity", "include_projected_gravity", "include_reference")
    disabled_flags = [name for name in required_observation_flags if not bool(observation.get(name, True))]
    if disabled_flags:
        raise RuntimeError(
            "native G1 observation contract requires enabled flags: " + ", ".join(disabled_flags)
        )
    action = config.get("action") if isinstance(config.get("action"), dict) else {}
    control = config.get("control") if isinstance(config.get("control"), dict) else {}
    decimation = int(control.get("decimation", 1))
    if decimation != 1:
        raise RuntimeError(
            "native G1 timing requires control.decimation=1 because the registered "
            f"G1 control_dt and policy_dt are both {G1_CONTROL_DT} seconds"
        )
    if str(control.get("kp_profile", "g1_default")) != "g1_default":
        raise RuntimeError("native G1 task only supports kp_profile=g1_default")
    if str(control.get("kd_profile", "g1_default")) != "g1_default":
        raise RuntimeError("native G1 task only supports kd_profile=g1_default")
    env_cfg.history_length = int(observation.get("history_length", 3))
    env_cfg.observation_clip = float(observation.get("clip_value", 100.0))
    env_cfg.observation_space = BASE_OBSERVATION_DIM * env_cfg.history_length + 10
    env_cfg.action_scale = float(action.get("scale", 0.25))
    env_cfg.action_clip = float(action.get("clip", 1.0))
    env_cfg.decimation = decimation
    env_cfg.sim.dt = G1_CONTROL_DT
    env_cfg.sim.render_interval = env_cfg.decimation
    randomization = config.get("domain_randomization") if isinstance(config.get("domain_randomization"), dict) else {}
    if bool(randomization.get("enabled", True)):
        events = G1DomainRandomizationCfg()
        mass_scale = tuple(float(value) for value in randomization.get("mass_scale", (0.95, 1.05)))
        friction = tuple(float(value) for value in randomization.get("friction", (0.7, 1.3)))
        motor_strength = tuple(float(value) for value in randomization.get("motor_strength", (0.95, 1.05)))
        for name, values in (("mass_scale", mass_scale), ("friction", friction), ("motor_strength", motor_strength)):
            if len(values) != 2 or values[0] <= 0 or values[1] < values[0]:
                raise RuntimeError(f"invalid G1 domain-randomization range for {name}: {values}")
        events.robot_mass.params["mass_distribution_params"] = mass_scale
        events.robot_material.params["static_friction_range"] = friction
        events.robot_material.params["dynamic_friction_range"] = friction
        # The platform's motor-strength range scales the position-controller
        # stiffness and damping together for implicit Isaac actuators.
        events.actuator_gains.params["stiffness_distribution_params"] = motor_strength
        events.actuator_gains.params["damping_distribution_params"] = motor_strength
        env_cfg.events = events
    else:
        env_cfg.events = None
    reward_config = config.get("reward_config") if isinstance(config.get("reward_config"), dict) else {}
    reward_terms = reward_config.get("terms") if isinstance(reward_config.get("terms"), list) else []
    env_cfg.reward_weights = {
        str(term["id"]): float(term["weight"])
        for term in reward_terms
        if isinstance(term, dict) and term.get("enabled", True) and "id" in term and "weight" in term
    }
    env_cfg.reward_sigmas = {
        str(term["id"]): float(term["params"]["sigma"])
        for term in reward_terms
        if isinstance(term, dict)
        and isinstance(term.get("params"), dict)
        and "sigma" in term["params"]
        and "id" in term
    }
    terminations = reward_config.get("terminations")
    if isinstance(terminations, list):
        env_cfg.terminations = tuple(str(value) for value in terminations)
    runner_cfg = G1MimicRunnerCfg()
    runner_cfg.seed = env_cfg.seed
    runner_cfg.device = device
    runner_cfg.max_iterations = int(ppo.get("max_iterations", runner_cfg.max_iterations))
    runner_cfg.num_steps_per_env = int(ppo.get("rollout_length", runner_cfg.num_steps_per_env))
    runner_cfg.policy.actor_hidden_dims = list(ppo.get("hidden_dims", runner_cfg.policy.actor_hidden_dims))
    runner_cfg.policy.critic_hidden_dims = list(ppo.get("hidden_dims", runner_cfg.policy.critic_hidden_dims))
    runner_cfg.algorithm.learning_rate = float(ppo.get("learning_rate", runner_cfg.algorithm.learning_rate))
    runner_cfg.algorithm.schedule = str(ppo.get("schedule", runner_cfg.algorithm.schedule))
    runner_cfg.algorithm.gamma = float(ppo.get("gamma", runner_cfg.algorithm.gamma))
    runner_cfg.algorithm.lam = float(ppo.get("lam", runner_cfg.algorithm.lam))
    runner_cfg.algorithm.clip_param = float(ppo.get("clip_param", runner_cfg.algorithm.clip_param))
    runner_cfg.algorithm.entropy_coef = float(ppo.get("entropy_coef", runner_cfg.algorithm.entropy_coef))
    runner_cfg.algorithm.value_loss_coef = float(ppo.get("value_loss_coef", runner_cfg.algorithm.value_loss_coef))
    runner_cfg.algorithm.max_grad_norm = float(ppo.get("max_grad_norm", runner_cfg.algorithm.max_grad_norm))
    runner_cfg.save_interval = max(1, min(runner_cfg.max_iterations, 100))
    return env_cfg, runner_cfg


def _open_runner(*, motion: Path, config: dict[str, Any], device: str, output_dir: Path, num_envs: int | None = None):
    env_cfg, runner_cfg = _configs(motion=motion, config=config, device=device, num_envs=num_envs)
    env = G1MimicEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(output_dir / "logs"), device=device)
    wrapped.seed(runner_cfg.seed)
    return wrapped, runner, runner_cfg


def _context_path(checkpoint: Path) -> Path:
    return checkpoint.resolve().parent / "g1_task_context.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embed_context(checkpoint: Path, context: dict[str, Any], motion: Path) -> None:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("RSL-RL checkpoint is not a dictionary and cannot carry platform context")
    payload[CHECKPOINT_CONTEXT_KEY] = {**context, "motion_npz": motion.read_bytes()}
    torch.save(payload, str(checkpoint))


def _load_context(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    context = payload.get(CHECKPOINT_CONTEXT_KEY) if isinstance(payload, dict) else None
    if not isinstance(context, dict):
        raise RuntimeError("G1 checkpoint does not contain embedded AllRobotRLLLab context")
    if context.get("schema_version") != "g1_native_task_context.v1" or context.get("task_id") != TASK_ID:
        raise RuntimeError("G1 checkpoint platform context is incompatible")
    expected_motion_hash = str(context.get("motion_sha256", ""))
    motion = Path(str(context.get("motion_path", ""))).expanduser().resolve()
    if not motion.is_file() or _file_sha256(motion) != expected_motion_hash:
        motion_bytes = context.get("motion_npz")
        if not isinstance(motion_bytes, bytes):
            raise RuntimeError("G1 checkpoint does not contain a recoverable TrainMotionNPZ")
        if hashlib.sha256(motion_bytes).hexdigest() != expected_motion_hash:
            raise RuntimeError("embedded G1 TrainMotionNPZ checksum does not match checkpoint context")
        motion = checkpoint.parent / f"train_motion-{expected_motion_hash[:16]}.npz"
        motion.write_bytes(motion_bytes)
    config = context.get("training_config")
    if not isinstance(config, dict):
        raise RuntimeError("G1 checkpoint context does not contain training_config")
    return motion, config


def train(*, task_id: str, manifest: Path, output_dir: Path, device: str) -> int:
    if task_id != TASK_ID:
        raise RuntimeError(f"unsupported native G1 task: {task_id}")
    motion, config, manifest_payload = _training_input(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    env, runner, runner_cfg = _open_runner(
        motion=motion,
        config=config,
        device=device,
        output_dir=output_dir,
    )
    try:
        runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
        checkpoint = output_dir / "checkpoint.pt"
        runner.save(str(checkpoint))
    finally:
        env.close()
    context = {
        "schema_version": "g1_native_task_context.v1",
        "task_id": task_id,
        "motion_path": str(motion),
        "motion_sha256": _file_sha256(motion),
        "training_config": config,
        "run_manifest": manifest_payload,
        "run_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint),
    }
    _embed_context(checkpoint, context, motion)
    _context_path(checkpoint).write_text(json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    metrics = [
        {
            "iteration": runner_cfg.max_iterations,
            "name": "train/iterations_completed",
            "value": float(runner_cfg.max_iterations),
            "num_envs": int(env.num_envs),
            "device": device,
            "run_id": manifest_payload.get("run_id"),
        }
    ]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _policy(runner: OnPolicyRunner):
    policy = getattr(runner.alg, "policy", None)
    if policy is None:
        policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        raise RuntimeError("RSL-RL runner does not expose an exportable policy")
    normalizer = getattr(policy, "actor_obs_normalizer", None)
    if normalizer is None:
        normalizer = getattr(policy, "student_obs_normalizer", None)
    return policy, normalizer


def export(*, task_id: str, checkpoint: Path, output_dir: Path, device: str) -> int:
    if task_id != TASK_ID:
        raise RuntimeError(f"unsupported native G1 task: {task_id}")
    checkpoint = checkpoint.resolve()
    motion, config = _load_context(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, runner, _runner_cfg = _open_runner(
        motion=motion,
        config=config,
        device=device,
        output_dir=output_dir,
        num_envs=1,
    )
    try:
        runner.load(str(checkpoint))
        policy, normalizer = _policy(runner)
        export_policy_as_jit(policy, normalizer=normalizer, path=str(output_dir), filename="policy.pt")
        export_policy_as_onnx(policy, normalizer=normalizer, path=str(output_dir), filename="policy.onnx")
    finally:
        env.close()
    files = {}
    for name in ("policy.pt", "policy.onnx"):
        path = output_dir / name
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
    (output_dir / "export_metadata.json").write_text(
        json.dumps({"schema_version": "g1_native_export.v1", "task_id": task_id, "files": files}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def play(*, task_id: str, checkpoint: Path, output_dir: Path, device: str) -> int:
    if task_id != TASK_ID:
        raise RuntimeError(f"unsupported native G1 task: {task_id}")
    checkpoint = checkpoint.resolve()
    motion, config = _load_context(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    env, runner, _runner_cfg = _open_runner(
        motion=motion,
        config=config,
        device=device,
        output_dir=output_dir,
        num_envs=1,
    )
    step_count = max(1, int(os.getenv("G1_PLAY_STEPS", "500")))
    total_reward = 0.0
    resets = 0
    try:
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=device)
        observations = env.get_observations()
        with torch.inference_mode():
            for _ in range(step_count):
                actions = policy(observations)
                observations, rewards, dones, _extras = env.step(actions)
                total_reward += float(rewards.mean().item())
                resets += int(dones.sum().item())
    finally:
        env.close()
    report = {
        "schema_version": "g1_native_play.v1",
        "task_id": task_id,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "device": device,
        "steps": step_count,
        "mean_reward": total_reward / step_count,
        "resets": resets,
    }
    (output_dir / "play_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


__all__ = ["G1MimicEnv", "G1MimicEnvCfg", "G1MimicRunnerCfg", "export", "play", "train"]
