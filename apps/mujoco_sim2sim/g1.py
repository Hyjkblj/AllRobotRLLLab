"""Evaluate the native G1 mimic policy in the upstream Unitree MuJoCo model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


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
    "pelvis", "torso_link", "left_rubber_hand", "right_rubber_hand",
    "left_ankle_roll_link", "right_ankle_roll_link",
)
MUJOCO_METRIC_BODY_NAMES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
CONTROL_DT = 0.02
ACTION_SCALE = 0.25
BASE_OBSERVATION_DIM = len(JOINT_NAMES) * 4 + 6
TRAIN_MOTION_FORMAT = "train_motion.v1"
ROBOT_ID = "unitree_g1_29dof"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQUENCY = 10.0 * 2.0 * np.pi
DAMPING_RATIO = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gain(armature: float) -> tuple[float, float]:
    stiffness = armature * NATURAL_FREQUENCY**2
    damping = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQUENCY
    return stiffness, damping


def _control_gains() -> tuple[np.ndarray, np.ndarray]:
    gains: list[tuple[float, float]] = []
    for name in JOINT_NAMES:
        if "hip_roll" in name or "knee" in name:
            gains.append(_gain(ARMATURE_7520_22))
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            gains.append(_gain(ARMATURE_7520_14))
        elif "ankle" in name or name in {"waist_roll_joint", "waist_pitch_joint"}:
            stiffness, damping = _gain(ARMATURE_5020)
            gains.append((2.0 * stiffness, 2.0 * damping))
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            gains.append(_gain(ARMATURE_4010))
        else:
            gains.append(_gain(ARMATURE_5020))
    return (
        np.asarray([value[0] for value in gains], dtype=np.float64),
        np.asarray([value[1] for value in gains], dtype=np.float64),
    )


class Policy:
    def __init__(self, path: Path, *, observation_dim: int):
        self.path = path.resolve()
        if self.path.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required to evaluate an ONNX policy") from exc
            self._session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            self._input = self._session.get_inputs()[0]
            input_dim = self._input.shape[-1]
            output_dim = self._session.get_outputs()[0].shape[-1]
            self.input_dim = int(input_dim) if isinstance(input_dim, int) else observation_dim
            self.output_dim = int(output_dim) if isinstance(output_dim, int) else len(JOINT_NAMES)
            self._kind = "onnx"
        elif self.path.suffix.lower() in {".pt", ".torchscript"}:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("torch is required to evaluate a TorchScript policy") from exc
            self._torch = torch
            self._model = torch.jit.load(str(self.path), map_location="cpu").eval()
            self.input_dim = observation_dim
            self.output_dim = len(JOINT_NAMES)
            self._kind = "torchscript"
        else:
            raise RuntimeError(f"unsupported policy format: {self.path.suffix}")

    def infer(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if self.input_dim < 0:
            self.input_dim = int(value.shape[1])
        if value.shape[1] != self.input_dim:
            raise RuntimeError(f"policy input is {self.input_dim}, evaluator produced {value.shape[1]}")
        if self._kind == "onnx":
            result = self._session.run(None, {self._input.name: value})[0]
        else:
            with self._torch.inference_mode():
                result = self._model(self._torch.from_numpy(value)).cpu().numpy()
        result = np.asarray(result, dtype=np.float64).reshape(-1)
        if result.shape != (len(JOINT_NAMES),) or not np.isfinite(result).all():
            raise RuntimeError(f"policy output must be finite with shape ({len(JOINT_NAMES)},)")
        return result


def _load_motion(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "format_version", "robot_id", "fps", "joint_names", "body_names",
            "coord_frame", "quat_convention", "source_motion_hash", "compiler_version",
            "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"TrainMotionNPZ is missing fields: {sorted(missing)}")
        scalar = lambda name: np.asarray(archive[name]).item()
        if str(scalar("format_version")) != TRAIN_MOTION_FORMAT:
            raise RuntimeError(f"TrainMotionNPZ format_version must be {TRAIN_MOTION_FORMAT}")
        if str(scalar("robot_id")) != ROBOT_ID:
            raise RuntimeError(f"TrainMotionNPZ robot_id must be {ROBOT_ID}")
        if str(scalar("coord_frame")) != "world_z_up" or str(scalar("quat_convention")) != "wxyz":
            raise RuntimeError("TrainMotionNPZ coordinate or quaternion convention is incompatible")
        if tuple(str(value) for value in np.asarray(archive["joint_names"]).tolist()) != JOINT_NAMES:
            raise RuntimeError("TrainMotionNPZ joint_names do not match G1 order")
        if tuple(str(value) for value in np.asarray(archive["body_names"]).tolist()) != BODY_NAMES:
            raise RuntimeError("TrainMotionNPZ body_names do not match G1 order")
        result = {
            "fps": float(np.asarray(archive["fps"]).item()),
            "joint_pos": np.asarray(archive["joint_pos"], dtype=np.float64),
            "joint_vel": np.asarray(archive["joint_vel"], dtype=np.float64),
            "body_pos_w": np.asarray(archive["body_pos_w"], dtype=np.float64),
            "body_quat_w": np.asarray(archive["body_quat_w"], dtype=np.float64),
            "body_lin_vel_w": np.asarray(archive["body_lin_vel_w"], dtype=np.float64),
            "body_ang_vel_w": np.asarray(archive["body_ang_vel_w"], dtype=np.float64),
        }
    frames = result["joint_pos"].shape[0] if result["joint_pos"].ndim == 2 else 0
    expected_shapes = {
        "joint_pos": (frames, len(JOINT_NAMES)),
        "joint_vel": (frames, len(JOINT_NAMES)),
        "body_pos_w": (frames, len(BODY_NAMES), 3),
        "body_quat_w": (frames, len(BODY_NAMES), 4),
        "body_lin_vel_w": (frames, len(BODY_NAMES), 3),
        "body_ang_vel_w": (frames, len(BODY_NAMES), 3),
    }
    if frames < 2 or not 15.0 <= result["fps"] <= 120.0:
        raise RuntimeError("TrainMotionNPZ must contain at least two frames at 15-120 fps")
    for name, shape in expected_shapes.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise RuntimeError(f"TrainMotionNPZ {name} must be finite with shape {shape}")
    quaternion_norms = np.linalg.norm(result["body_quat_w"], axis=-1)
    if float(np.max(np.abs(quaternion_norms - 1.0))) > 1.0e-3:
        raise RuntimeError("TrainMotionNPZ body_quat_w is not normalized")
    return result


def _resample_motion(motion: dict[str, Any]) -> dict[str, Any]:
    source_fps = float(motion["fps"])
    target_fps = 1.0 / CONTROL_DT
    frames = int(motion["joint_pos"].shape[0])
    if frames < 2 or abs(source_fps - target_fps) < 1.0e-6:
        return motion
    duration = (frames - 1) / source_fps
    count = max(2, int(round(duration * target_fps)) + 1)
    positions = np.linspace(0.0, frames - 1, count)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, frames - 1)
    alpha = positions - lower
    result: dict[str, Any] = {"fps": target_fps}
    for name, source in motion.items():
        if name == "fps":
            continue
        blend = alpha.reshape((count,) + (1,) * (source.ndim - 1))
        if name != "body_quat_w":
            result[name] = source[lower] * (1.0 - blend) + source[upper] * blend
            continue
        first = source[lower]
        second = source[upper]
        second = np.where(np.sum(first * second, axis=-1, keepdims=True) < 0.0, -second, second)
        quaternions = first * (1.0 - blend) + second * blend
        result[name] = quaternions / np.maximum(np.linalg.norm(quaternions, axis=-1, keepdims=True), 1.0e-12)
    return result


def _quat_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _write_png(path: Path, pixels: np.ndarray) -> None:
    image = np.asarray(pixels, dtype=np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("PNG writer expects RGB pixels")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def evaluate(
    *,
    seed: int,
    policy_path: Path,
    motion_path: Path,
    model_path: Path,
    output_dir: Path,
    max_steps: int | None = None,
    render_final: bool = False,
    observation_dim: int = BASE_OBSERVATION_DIM * 3 + 10,
) -> dict[str, float]:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("mujoco is required by the G1 sim2sim evaluator") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    policy = Policy(policy_path, observation_dim=observation_dim)
    motion = _resample_motion(_load_motion(motion_path))
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    if abs(round(CONTROL_DT / model.opt.timestep) * model.opt.timestep - CONTROL_DT) > 1.0e-9:
        raise RuntimeError(f"MuJoCo timestep {model.opt.timestep} does not divide control_dt {CONTROL_DT}")
    substeps = int(round(CONTROL_DT / model.opt.timestep))

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in MUJOCO_METRIC_BODY_NAMES]
    if min(joint_ids) < 0 or min(body_ids) < 0:
        raise RuntimeError("Unitree MuJoCo model does not satisfy the G1 RobotSpec names")
    qpos_ids = np.asarray([model.jnt_qposadr[index] for index in joint_ids], dtype=np.int64)
    qvel_ids = np.asarray([model.jnt_dofadr[index] for index in joint_ids], dtype=np.int64)
    actuator_ids = []
    for joint_id in joint_ids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(matches) != 1:
            raise RuntimeError(f"expected one actuator for MuJoCo joint id {joint_id}, got {len(matches)}")
        actuator_ids.append(int(matches[0]))
    actuator_ids = np.asarray(actuator_ids, dtype=np.int64)
    torque_limits = np.max(np.abs(model.actuator_ctrlrange[actuator_ids]), axis=1)
    kp, kd = _control_gains()

    rng = np.random.default_rng(seed)
    model.geom_friction[:, :2] *= rng.uniform(0.95, 1.05)
    data.qpos[:3] = motion["body_pos_w"][0, 0]
    initial_quaternion = motion["body_quat_w"][0, 0]
    data.qpos[3:7] = initial_quaternion / max(float(np.linalg.norm(initial_quaternion)), 1.0e-12)
    data.qpos[:2] += rng.normal(0.0, 0.002, size=2)
    data.qpos[qpos_ids] = motion["joint_pos"][0] + rng.normal(0.0, 0.002, size=len(JOINT_NAMES))
    data.qvel[:3] = motion["body_lin_vel_w"][0, 0]
    data.qvel[3:6] = motion["body_ang_vel_w"][0, 0]
    data.qvel[qvel_ids] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)

    if (policy.input_dim - 10) % BASE_OBSERVATION_DIM != 0:
        raise RuntimeError(f"policy input dimension {policy.input_dim} is incompatible with the G1 observation contract")
    history_length = (policy.input_dim - 10) // BASE_OBSERVATION_DIM
    if history_length < 1:
        raise RuntimeError("policy observation history must contain at least one frame")
    history = np.zeros((history_length, BASE_OBSERVATION_DIM), dtype=np.float32)
    total_steps = min(int(motion["joint_pos"].shape[0]) - 1, max_steps or 10**9)
    if total_steps < 1:
        raise RuntimeError("sim2sim motion must contain at least two frames")

    joint_squared: list[float] = []
    root_squared: list[float] = []
    orientation_errors: list[float] = []
    saturation_samples = 0
    saturation_total = 0
    foot_slips: list[float] = []
    rows: list[dict[str, float | int]] = []
    failure = ""

    for step in range(1, total_steps + 1):
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_ids[0], velocity, 1)
        current = np.concatenate(
            (
                data.qpos[qpos_ids], data.qvel[qvel_ids], velocity[3:], velocity[:3],
                motion["joint_pos"][step], motion["joint_vel"][step],
            )
        ).astype(np.float32)
        history[1:] = history[:-1]
        history[0] = current
        phase = step / float(motion["joint_pos"].shape[0])
        observation = np.concatenate(
            (
                history.reshape(-1), motion["body_pos_w"][step, 0], motion["body_quat_w"][step, 0],
                np.asarray([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase), phase]),
            )
        )
        action = np.clip(policy.infer(observation), -1.0, 1.0)
        target = motion["joint_pos"][step] + ACTION_SCALE * action
        torque = kp * (target - data.qpos[qpos_ids]) - kd * data.qvel[qvel_ids]
        clipped = np.clip(torque, -torque_limits, torque_limits)
        saturation_samples += int(np.count_nonzero(np.abs(torque) >= torque_limits))
        saturation_total += len(torque)
        data.ctrl[actuator_ids] = clipped
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        joint_rmse = float(np.sqrt(np.mean(np.square(data.qpos[qpos_ids] - motion["joint_pos"][step]))))
        root_error = float(np.linalg.norm(data.qpos[:3] - motion["body_pos_w"][step, 0]))
        orientation_error = _quat_error_deg(data.qpos[3:7], motion["body_quat_w"][step, 0])
        slip_values = []
        for body_id in body_ids[-2:]:
            foot_velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, foot_velocity, 0)
            if data.xpos[body_id, 2] < 0.08:
                slip_values.append(float(np.linalg.norm(foot_velocity[3:5])))
        foot_slip = float(np.mean(slip_values)) if slip_values else 0.0
        joint_squared.append(joint_rmse**2)
        root_squared.append(root_error**2)
        orientation_errors.append(orientation_error)
        foot_slips.append(foot_slip)
        rows.append(
            {
                "step": step, "time_s": step * CONTROL_DT, "joint_rmse_rad": joint_rmse,
                "root_position_error_m": root_error, "orientation_error_deg": orientation_error,
                "saturation_ratio": saturation_samples / saturation_total, "foot_slip_mps": foot_slip,
            }
        )
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            failure = "non_finite_state"
        elif root_error > 0.40:
            failure = "root_position_error"
        elif orientation_error > 50.0:
            failure = "root_orientation_error"
        if failure:
            break

    completed = len(rows)
    metrics = {
        "survival_rate": completed / total_steps,
        "joint_rmse_rad": float(np.sqrt(np.mean(joint_squared))),
        "root_position_rmse_m": float(np.sqrt(np.mean(root_squared))),
        "orientation_error_deg": float(np.mean(orientation_errors)),
        "saturation_ratio": saturation_samples / max(1, saturation_total),
        "foot_slip_mps": float(np.mean(foot_slips)),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "trace.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "step", "time_s", "joint_rmse_rad", "root_position_error_m",
                "orientation_error_deg", "saturation_ratio", "foot_slip_mps",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "g1_mujoco_seed.v1", "seed": seed, "failure": failure or None,
        "completed_steps": completed, "requested_steps": total_steps, "control_dt": CONTROL_DT,
        "physics_dt": float(model.opt.timestep), "substeps": substeps,
        "policy": {"path": str(policy_path.resolve()), "sha256": _sha256(policy_path)},
        "motion": {"path": str(motion_path.resolve()), "sha256": _sha256(motion_path)},
        "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
        "metrics": metrics,
    }
    (output_dir / "seed_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if render_final:
        try:
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                renderer.update_scene(data)
                _write_png(output_dir / "final_frame.png", renderer.render())
        except Exception as exc:
            (output_dir / "render_error.txt").write_text(str(exc) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--unitree-mujoco-root", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--observation-dim", type=int, default=BASE_OBSERVATION_DIM * 3 + 10)
    parser.add_argument("--render-final", action="store_true")
    args = parser.parse_args(argv)
    root = args.unitree_mujoco_root or Path(os.getenv("UNITREE_MUJOCO_PATH", ""))
    model = args.model or root / "unitree_robots" / "g1" / "scene_29dof.xml"
    if not model.is_file():
        parser.error(f"Unitree G1 MuJoCo scene does not exist: {model}")
    evaluate(
        seed=args.seed,
        policy_path=args.policy.resolve(),
        motion_path=args.motion.resolve(),
        model_path=model.resolve(),
        output_dir=args.output.resolve(),
        max_steps=args.max_steps,
        render_final=args.render_final,
        observation_dim=args.observation_dim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ACTION_SCALE", "BASE_OBSERVATION_DIM", "BODY_NAMES", "CONTROL_DT", "JOINT_NAMES", "MUJOCO_METRIC_BODY_NAMES", "ROBOT_ID", "TRAIN_MOTION_FORMAT", "evaluate", "main"]
