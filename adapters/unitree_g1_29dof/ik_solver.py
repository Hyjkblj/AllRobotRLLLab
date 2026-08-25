"""CPU MuJoCo constrained IK for G1 body-position targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.app.domain.motion import MotionArrays


class G1MuJoCoIKSolver:
    """Damped least-squares position IK with G1 joint-limit clamping.

    MuJoCo is imported only when ``apply`` is called so the API can still load
    contract endpoints in environments that do not contain the renderer wheel.
    ``target_offset`` is interpreted relative to the body's position at the
    first frame of each requested range.
    """

    def __init__(self, *, repository_root: Path | None = None) -> None:
        self.repository_root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
        self.xml_path = self.repository_root / "third_party" / "GMR-master" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"

    @staticmethod
    def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
        return np.asarray([quaternion[3], quaternion[0], quaternion[1], quaternion[2]], dtype=np.float64)

    @staticmethod
    def _wxyz_to_xyzw(quaternion: np.ndarray) -> np.ndarray:
        return np.asarray([quaternion[1], quaternion[2], quaternion[3], quaternion[0]], dtype=np.float64)

    def apply(self, arrays: MotionArrays, targets: list[dict]) -> MotionArrays:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError("MuJoCo runtime is not installed for the G1 IK worker") from exc
        if not self.xml_path.is_file():
            raise RuntimeError(f"G1 IK model not found: {self.xml_path}")
        model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        data = mujoco.MjData(model)
        result = arrays.copy()
        joint_ids = {}
        for name in result.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"G1 IK joint not found in MuJoCo model: {name}")
            joint_ids[name] = joint_id

        for target_index, target in enumerate(targets):
            body_name = target.get("body_name")
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"G1 IK body not found in MuJoCo model: {body_name}")
            start = int(target.get("frame_start", 0))
            end = int(target.get("frame_end", len(result.joint_pos) - 1))
            offset = np.asarray(target.get("target_offset", []), dtype=np.float64)
            if offset.shape != (3,):
                raise ValueError(f"IK target {target_index} target_offset must have shape [3]")
            if start < 0 or end >= len(result.joint_pos) or end < start:
                raise ValueError(f"IK target {target_index} frame range is invalid")

            for frame in range(start, end + 1):
                data.qpos[:] = 0.0
                data.qpos[:3] = result.root_pos[frame]
                data.qpos[3:7] = self._xyzw_to_wxyz(result.root_rot[frame])
                for name, joint_id in joint_ids.items():
                    data.qpos[model.jnt_qposadr[joint_id]] = result.joint_pos[frame, result.joint_names.index(name)]
                mujoco.mj_forward(model, data)
                target_position = data.xpos[body_id].copy() + offset
                jacp = np.zeros((3, model.nv), dtype=np.float64)
                jacr = np.zeros((3, model.nv), dtype=np.float64)
                dof_addresses = [model.jnt_dofadr[joint_ids[name]] for name in result.joint_names]
                qpos_addresses = [model.jnt_qposadr[joint_ids[name]] for name in result.joint_names]
                for _ in range(20):
                    mujoco.mj_forward(model, data)
                    error = target_position - data.xpos[body_id]
                    if float(np.linalg.norm(error)) < 1e-4:
                        break
                    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
                    jacobian = jacp[:, dof_addresses]
                    damping = 1e-4 * np.eye(3)
                    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
                    data.qpos[qpos_addresses] += delta
                    for name, joint_id in joint_ids.items():
                        qpos_address = model.jnt_qposadr[joint_id]
                        data.qpos[qpos_address] = np.clip(data.qpos[qpos_address], model.jnt_range[joint_id, 0], model.jnt_range[joint_id, 1])
                residual = float(np.linalg.norm(target_position - data.xpos[body_id]))
                if residual > 0.02:
                    raise RuntimeError(f"G1 IK residual is too large for {body_name}: {residual:.5f} m")
                for name, joint_id in joint_ids.items():
                    result.joint_pos[frame, result.joint_names.index(name)] = data.qpos[model.jnt_qposadr[joint_id]]

        return result


__all__ = ["G1MuJoCoIKSolver"]
