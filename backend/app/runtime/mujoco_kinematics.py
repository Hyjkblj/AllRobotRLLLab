"""MuJoCo-backed body kinematics compiler for TrainMotionNPZ."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from backend.app.runtime.contracts import RunnerError


def _quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    value = value / np.maximum(norm, 1e-9)
    return value[..., [3, 0, 1, 2]]


def _quat_angular_velocity(quat_xyzw: np.ndarray, fps: float) -> np.ndarray:
    """Finite-difference angular velocity in world coordinates."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    if len(q) < 2:
        return np.zeros((len(q), 3), dtype=np.float64)
    # Small-angle derivative is robust for mocap rates and avoids another
    # dependency on scipy in the platform process.
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    derivative = np.gradient(q, 1.0 / fps, axis=0)
    angular = 2.0 * np.stack((derivative[:, 0], derivative[:, 1], derivative[:, 2]), axis=1)
    return angular


class MuJoCoKinematicsCompiler:
    name = "mujoco-kinematics-compiler"
    version = "mujoco-kinematics.v2"

    def __init__(self, *, model_path: Path, body_names: Iterable[str], joint_names: Iterable[str], allow_approximation: bool = False) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.body_names = tuple(body_names)
        self.joint_names = tuple(joint_names)
        self.allow_approximation = allow_approximation

    def compile(self, *, joint_pos: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray, fps: float) -> dict[str, np.ndarray | str | float]:
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_rot = np.asarray(root_rot, dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(self.joint_names):
            raise RunnerError("KINEMATICS_INPUT_INVALID", "joint_pos shape does not match the locked robot contract")
        if root_pos.shape != (len(joint_pos), 3) or root_rot.shape != (len(joint_pos), 4):
            raise RunnerError("KINEMATICS_INPUT_INVALID", "root arrays do not match joint_pos frame count")
        if not self.model_path.is_file():
            if self.allow_approximation:
                return self._approximate(joint_pos, root_pos, root_rot, fps)
            raise RunnerError("KINEMATICS_MODEL_NOT_FOUND", f"MuJoCo model does not exist: {self.model_path}")
        try:
            import mujoco
        except ImportError as exc:
            if self.allow_approximation:
                return self._approximate(joint_pos, root_pos, root_rot, fps)
            raise RunnerError("KINEMATICS_RUNTIME_UNAVAILABLE", "mujoco is not installed in the selected runtime") from exc
        try:
            model = mujoco.MjModel.from_xml_path(str(self.model_path))
            data = mujoco.MjData(model)
        except Exception as exc:
            raise RunnerError("KINEMATICS_MODEL_INVALID", f"unable to load MuJoCo model: {exc}") from exc
        body_ids = []
        for name in self.body_names:
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            if body_id < 0:
                raise RunnerError("KINEMATICS_BODY_NOT_FOUND", f"body '{name}' is not present in MuJoCo model")
            body_ids.append(body_id)
        qpos_adrs: list[int] = []
        qvel_adrs: list[int] = []
        for name in self.joint_names:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if joint_id < 0:
                raise RunnerError("KINEMATICS_JOINT_NOT_FOUND", f"joint '{name}' is not present in MuJoCo model")
            qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
            qvel_adrs.append(int(model.jnt_dofadr[joint_id]))
        frame_count = len(joint_pos)
        joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0) if frame_count > 1 else np.zeros_like(joint_pos)
        root_lin_vel = np.gradient(root_pos, 1.0 / fps, axis=0) if frame_count > 1 else np.zeros_like(root_pos)
        root_ang_vel = _quat_angular_velocity(root_rot, fps)
        body_pos = np.zeros((frame_count, len(body_ids), 3), dtype=np.float32)
        body_quat = np.zeros((frame_count, len(body_ids), 4), dtype=np.float32)
        body_lin_vel = np.zeros_like(body_pos)
        body_ang_vel = np.zeros_like(body_pos)
        for frame in range(frame_count):
            data.qpos[:] = 0.0
            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = _quat_xyzw_to_wxyz(root_rot[frame])
            for index, address in enumerate(qpos_adrs):
                if address < model.nq:
                    data.qpos[address] = joint_pos[frame, index]
            data.qvel[:] = 0.0
            if model.nv >= 6:
                data.qvel[:3] = root_lin_vel[frame]
                data.qvel[3:6] = root_ang_vel[frame]
            for index, address in enumerate(qvel_adrs):
                if address < model.nv:
                    data.qvel[address] = joint_vel[frame, index]
            mujoco.mj_forward(model, data)
            body_pos[frame] = data.xpos[body_ids]
            body_quat[frame] = data.xquat[body_ids]
            # MuJoCo cvel uses angular velocity followed by linear velocity.
            body_ang_vel[frame] = data.cvel[body_ids, :3]
            body_lin_vel[frame] = data.cvel[body_ids, 3:]
        return {"joint_pos": joint_pos.astype(np.float32), "joint_vel": joint_vel.astype(np.float32), "body_pos_w": body_pos, "body_quat_w": body_quat, "body_lin_vel_w": body_lin_vel, "body_ang_vel_w": body_ang_vel, "fps": float(fps), "compiler_version": self.version}

    def _approximate(self, joint_pos: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray, fps: float) -> dict[str, np.ndarray | str | float]:
        count = len(joint_pos)
        names = len(self.body_names)
        joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0) if count > 1 else np.zeros_like(joint_pos)
        root_vel = np.gradient(root_pos, 1.0 / fps, axis=0) if count > 1 else np.zeros_like(root_pos)
        return {"joint_pos": joint_pos.astype(np.float32), "joint_vel": joint_vel.astype(np.float32), "body_pos_w": np.repeat(root_pos[:, None, :], names, axis=1).astype(np.float32), "body_quat_w": np.repeat(_quat_xyzw_to_wxyz(root_rot)[:, None, :], names, axis=1).astype(np.float32), "body_lin_vel_w": np.repeat(root_vel[:, None, :], names, axis=1).astype(np.float32), "body_ang_vel_w": np.zeros((count, names, 3), dtype=np.float32), "fps": float(fps), "compiler_version": f"{self.version}+approximation"}


__all__ = ["MuJoCoKinematicsCompiler"]
