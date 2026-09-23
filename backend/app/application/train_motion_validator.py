"""Validate the serialized TrainMotionNPZ contract before GPU execution."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from backend.app.domain.contracts import RobotSpec, SchemaVersion


class TrainMotionFileError(ValueError):
    """Raised when a serialized training motion violates the platform contract."""


_REQUIRED_ARRAYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}
_REQUIRED_METADATA = {
    "format_version",
    "robot_id",
    "fps",
    "joint_names",
    "body_names",
    "coord_frame",
    "quat_convention",
    "source_motion_hash",
    "compiler_version",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _scalar(value: np.ndarray, *, name: str) -> object:
    array = np.asarray(value)
    if array.shape != ():
        raise TrainMotionFileError(f"{name} must be a scalar")
    return array.item()


def _string_scalar(value: np.ndarray, *, name: str) -> str:
    result = _scalar(value, name=name)
    if not isinstance(result, str) or not result.strip():
        raise TrainMotionFileError(f"{name} must be a non-empty string")
    return result.strip()


def validate_train_motion_file(path: Path, robot: RobotSpec, *, expected_sha256: str | None = None) -> None:
    """Validate the exact bytes passed to the trainer against RobotSpec."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise TrainMotionFileError(f"TrainMotionNPZ does not exist: {source}")
    if expected_sha256 is not None:
        if not _SHA256.fullmatch(expected_sha256):
            raise TrainMotionFileError("expected TrainMotionNPZ hash must be a lowercase SHA-256")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise TrainMotionFileError("TrainMotionNPZ bytes do not match the Run Manifest hash")
    try:
        with np.load(source, allow_pickle=False) as archive:
            names = set(archive.files)
            missing = (_REQUIRED_ARRAYS | _REQUIRED_METADATA).difference(names)
            if missing:
                raise TrainMotionFileError(f"missing fields: {sorted(missing)}")

            format_version = _string_scalar(archive["format_version"], name="format_version")
            robot_id = _string_scalar(archive["robot_id"], name="robot_id")
            coord_frame = _string_scalar(archive["coord_frame"], name="coord_frame")
            quat_convention = _string_scalar(archive["quat_convention"], name="quat_convention")
            source_hash = _string_scalar(archive["source_motion_hash"], name="source_motion_hash")
            _string_scalar(archive["compiler_version"], name="compiler_version")
            try:
                fps = float(_scalar(archive["fps"], name="fps"))
            except (TypeError, ValueError) as exc:
                raise TrainMotionFileError("fps must be numeric") from exc

            if format_version != SchemaVersion.TRAIN_MOTION.value:
                raise TrainMotionFileError(f"format_version must be {SchemaVersion.TRAIN_MOTION.value}")
            if robot_id != robot.robot_id:
                raise TrainMotionFileError(f"robot_id {robot_id!r} does not match {robot.robot_id!r}")
            if not 15.0 <= fps <= 120.0:
                raise TrainMotionFileError("fps must be between 15 and 120")
            if coord_frame != "world_z_up":
                raise TrainMotionFileError("coord_frame must be world_z_up")
            if quat_convention != "wxyz":
                raise TrainMotionFileError("quat_convention must be wxyz")
            if not _SHA256.fullmatch(source_hash):
                raise TrainMotionFileError("source_motion_hash must be a lowercase SHA-256")

            joint_names = tuple(str(item) for item in np.asarray(archive["joint_names"]).tolist())
            body_names = tuple(str(item) for item in np.asarray(archive["body_names"]).tolist())
            expected_joints = tuple(robot.joint_names)
            expected_bodies = tuple(robot.body_names)
            if joint_names != expected_joints:
                raise TrainMotionFileError("joint_names do not match RobotSpec order")
            if body_names != expected_bodies:
                raise TrainMotionFileError("body_names do not match RobotSpec order")

            arrays = {name: np.asarray(archive[name]) for name in _REQUIRED_ARRAYS}
            frame_count = arrays["joint_pos"].shape[0] if arrays["joint_pos"].ndim == 2 else 0
            body_count = len(expected_bodies)
            expected_shapes = {
                "joint_pos": (frame_count, robot.dof),
                "joint_vel": (frame_count, robot.dof),
                "body_pos_w": (frame_count, body_count, 3),
                "body_quat_w": (frame_count, body_count, 4),
                "body_lin_vel_w": (frame_count, body_count, 3),
                "body_ang_vel_w": (frame_count, body_count, 3),
            }
            if frame_count < 1:
                raise TrainMotionFileError("joint_pos must contain at least one frame")
            for name, expected in expected_shapes.items():
                value = arrays[name]
                if value.shape != expected:
                    raise TrainMotionFileError(f"{name} shape {value.shape} does not match {expected}")
                if not np.issubdtype(value.dtype, np.number):
                    raise TrainMotionFileError(f"{name} must be numeric")
                if not np.isfinite(value).all():
                    raise TrainMotionFileError(f"{name} contains non-finite values")

            quaternion_norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
            max_error = float(np.max(np.abs(quaternion_norms - 1.0)))
            if max_error > 1e-3:
                raise TrainMotionFileError(f"body_quat_w is not normalized (max error {max_error:.6g})")
    except TrainMotionFileError:
        raise
    except (OSError, ValueError) as exc:
        raise TrainMotionFileError(f"unable to read TrainMotionNPZ: {exc}") from exc


__all__ = ["TrainMotionFileError", "validate_train_motion_file"]
