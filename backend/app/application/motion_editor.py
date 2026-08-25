"""Server-authoritative motion editing and quality gates.

The browser may preview temporary poses, but only this service can produce an
edited trajectory eligible for compilation.  It operates on NumPy arrays and
RobotSpec metadata; persistence and HTTP concerns stay outside the service.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from backend.app.domain.contracts import (
    MotionEditConfig,
    MotionEditVersion,
    MotionQualityReport,
    RobotSpec,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from backend.app.domain.motion import MotionArrays


COMPILER_VERSION = "motion-editor.v1"


@dataclass(frozen=True)
class MotionEditResult:
    arrays: MotionArrays | None
    validation: ValidationResult
    quality: MotionQualityReport


class IKSolver(Protocol):
    def apply(self, arrays: MotionArrays, targets: list[dict]) -> MotionArrays: ...


def _issue(code: str, message: str, *, field: str | None = None, actual: object | None = None, expected: object | None = None, severity: ValidationSeverity = ValidationSeverity.BLOCKING_ERROR) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field, actual=actual, expected=expected, severity=severity)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_quaternions(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(quaternions, axis=1)
    invalid = ~np.isfinite(norms) | (norms <= 1e-8)
    normalized = np.zeros_like(quaternions)
    valid = ~invalid
    normalized[valid] = quaternions[valid] / norms[valid, None]
    return normalized, invalid


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.moveaxis(left, -1, 0)
    rx, ry, rz, rw = np.moveaxis(right, -1, 0)
    return np.stack((lw * rx + lx * rw + ly * rz - lz * ry,
                     lw * ry - lx * rz + ly * rw + lz * rx,
                     lw * rz + lx * ry - ly * rx + lz * rw,
                     lw * rw - lx * rx - ly * ry - lz * rz), axis=-1)


def _slerp_xyzw(source: np.ndarray, sample_times: np.ndarray, fps: float) -> np.ndarray:
    """SLERP a sequence at arbitrary source-frame times."""

    if len(source) == 1:
        return np.repeat(source, len(sample_times), axis=0)
    frame_positions = np.clip(sample_times * fps, 0.0, len(source) - 1)
    lower = np.floor(frame_positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = (frame_positions - lower)[:, None]
    first = source[lower].copy()
    second = source[upper].copy()
    dots = np.sum(first * second, axis=1)
    flip = dots < 0
    second[flip] *= -1
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    near = dots > 0.9995
    result = np.empty_like(first)
    result[near] = first[near] + alpha[near] * (second[near] - first[near])
    theta = np.arccos(dots[~near])
    sin_theta = np.sin(theta)
    weight_first = np.sin((1 - alpha[~near, 0]) * theta) / sin_theta
    weight_second = np.sin(alpha[~near, 0] * theta) / sin_theta
    result[~near] = weight_first[:, None] * first[~near] + weight_second[:, None] * second[~near]
    normalized, invalid = _normalize_quaternions(result)
    if invalid.any():
        raise ValueError("quaternion interpolation produced a zero quaternion")
    return normalized


def _resample(arrays: MotionArrays, time_scale: float) -> MotionArrays:
    if math.isclose(time_scale, 1.0, abs_tol=1e-9):
        return arrays
    source_frames = len(arrays.joint_pos)
    duration = (source_frames - 1) / arrays.fps
    output_frames = max(2, int(round(duration * time_scale * arrays.fps)) + 1)
    sample_times = np.arange(output_frames, dtype=np.float64) / arrays.fps / time_scale
    sample_times[-1] = duration
    source_times = np.arange(source_frames, dtype=np.float64) / arrays.fps
    joint_pos = np.column_stack([np.interp(sample_times, source_times, arrays.joint_pos[:, index]) for index in range(arrays.joint_pos.shape[1])])
    root_pos = np.column_stack([np.interp(sample_times, source_times, arrays.root_pos[:, index]) for index in range(3)])
    root_rot = _slerp_xyzw(arrays.root_rot, sample_times, arrays.fps)
    return MotionArrays(arrays.fps, joint_pos, root_pos, root_rot, arrays.joint_names, arrays.quat_convention, arrays.coord_frame)


def _apply_keyframes(joint_pos: np.ndarray, keyframes: list[dict], frame_count: int) -> tuple[np.ndarray, list[ValidationIssue]]:
    output = joint_pos.copy()
    issues: list[ValidationIssue] = []
    points: list[tuple[int, np.ndarray]] = []
    for index, keyframe in enumerate(keyframes):
        frame = keyframe.get("frame")
        values = keyframe.get("qpos", keyframe.get("joint_pos"))
        if not isinstance(frame, int) or frame < 0 or frame >= frame_count:
            issues.append(_issue("KEYFRAME_OUT_OF_RANGE", "keyframe frame is outside the motion", field=f"keyframes[{index}].frame", actual=frame, expected=[0, frame_count - 1]))
            continue
        if values is None:
            issues.append(_issue("KEYFRAME_DATA_MISSING", "keyframe must contain qpos or joint_pos", field=f"keyframes[{index}]"))
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (joint_pos.shape[1],):
            issues.append(_issue("KEYFRAME_SHAPE_INVALID", "keyframe pose must cover every G1 joint", field=f"keyframes[{index}]", actual=list(array.shape), expected=[joint_pos.shape[1]]))
            continue
        points.append((frame, array))
    points.sort(key=lambda item: item[0])
    for frame, values in points:
        output[frame] = values
    for (left_frame, left_values), (right_frame, right_values) in zip(points, points[1:]):
        if right_frame - left_frame > 1:
            weights = np.linspace(0.0, 1.0, right_frame - left_frame + 1)[:, None]
            output[left_frame : right_frame + 1] = left_values + weights * (right_values - left_values)
    return output, issues


class MotionEditor:
    """Apply a validated edit config and return a quality-gated trajectory."""

    def __init__(self, robot: RobotSpec, *, ik_solver: IKSolver | None = None, max_velocity_rad_s: float = 50.0) -> None:
        self.robot = robot
        self.ik_solver = ik_solver
        self.max_velocity_rad_s = max_velocity_rad_s

    def apply(self, arrays: MotionArrays, config: MotionEditConfig) -> MotionEditResult:
        issues: list[ValidationIssue] = []
        if config.robot_id != self.robot.robot_id:
            issues.append(_issue("ROBOT_ID_MISMATCH", "edit robot does not match the selected adapter", field="robot_id", actual=config.robot_id, expected=self.robot.robot_id))
        if arrays.quat_convention != "xyzw":
            issues.append(_issue("QUATERNION_CONVENTION_UNSUPPORTED", "motion editor expects external xyzw quaternions", field="quat_convention", actual=arrays.quat_convention, expected="xyzw"))
        if arrays.joint_pos.ndim != 2 or arrays.joint_pos.shape[1] != self.robot.dof:
            issues.append(_issue("MOTION_DOF_MISMATCH", "joint_pos must have the G1 29 DoF shape", field="joint_pos.shape", actual=list(arrays.joint_pos.shape), expected=["N", self.robot.dof]))
        if tuple(arrays.joint_names) != tuple(self.robot.joint_names):
            issues.append(_issue("MOTION_JOINT_ORDER_MISMATCH", "joint order does not match RobotSpec", field="joint_names", actual=list(arrays.joint_names), expected=self.robot.joint_names))
        if arrays.root_pos.shape != (len(arrays.joint_pos), 3) or arrays.root_rot.shape != (len(arrays.joint_pos), 4):
            issues.append(_issue("ROOT_SHAPE_INVALID", "root_pos/root_rot must align with joint frames", field="root_pos/root_rot.shape"))
        if not np.isfinite(arrays.joint_pos).all() or not np.isfinite(arrays.root_pos).all() or not np.isfinite(arrays.root_rot).all():
            issues.append(_issue("NONFINITE_VALUE", "motion arrays contain NaN or Inf"))
        if issues:
            return self._result(None, issues, arrays)

        edited = arrays.copy()
        normalized, invalid_quaternions = _normalize_quaternions(edited.root_rot)
        if invalid_quaternions.any():
            issues.append(_issue("ZERO_QUATERNION", "root rotation contains a zero quaternion", field="root_rot"))
            return self._result(None, issues, edited)
        edited = MotionArrays(edited.fps, edited.joint_pos, edited.root_pos, normalized, edited.joint_names, edited.quat_convention, edited.coord_frame)

        transform = config.global_transform
        if transform.translation != (0.0, 0.0, 0.0):
            edited.root_pos += np.asarray(transform.translation, dtype=np.float64)
        if not math.isclose(transform.yaw_offset, 0.0, abs_tol=1e-12):
            half = transform.yaw_offset / 2
            yaw = np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)
            edited.root_pos = edited.root_pos @ np.array([[math.cos(transform.yaw_offset), -math.sin(transform.yaw_offset), 0], [math.sin(transform.yaw_offset), math.cos(transform.yaw_offset), 0], [0, 0, 1]], dtype=np.float64).T
            edited.root_rot = _quat_multiply_xyzw(np.repeat(yaw[None, :], len(edited.root_rot), axis=0), edited.root_rot)
            edited.root_rot, _ = _normalize_quaternions(edited.root_rot)

        for index, offset in enumerate(config.joint_offsets):
            if offset.joint_name not in self.robot.joint_names:
                issues.append(_issue("JOINT_NOT_FOUND", "joint offset references an unknown G1 joint", field=f"joint_offsets[{index}].joint_name", actual=offset.joint_name))
                continue
            if offset.frame_start >= len(edited.joint_pos):
                issues.append(_issue("JOINT_OFFSET_OUT_OF_RANGE", "joint offset starts outside the motion", field=f"joint_offsets[{index}].frame_start", actual=offset.frame_start, expected=[0, len(edited.joint_pos) - 1]))
                continue
            if offset.frame_end >= len(edited.joint_pos):
                issues.append(_issue("JOINT_OFFSET_OUT_OF_RANGE", "joint offset ends outside the motion", field=f"joint_offsets[{index}].frame_end", actual=offset.frame_end, expected=[0, len(edited.joint_pos) - 1]))
                continue
            end = min(offset.frame_end, len(edited.joint_pos) - 1)
            joint_index = self.robot.joint_names.index(offset.joint_name)
            edited.joint_pos[offset.frame_start : end + 1, joint_index] += offset.position_offset

        edited.joint_pos, keyframe_issues = _apply_keyframes(edited.joint_pos, config.keyframes, len(edited.joint_pos))
        issues.extend(keyframe_issues)
        if config.ik_targets:
            for index, target in enumerate(config.ik_targets):
                body_name = target.get("body_name")
                if body_name not in self.robot.body_names:
                    issues.append(_issue("IK_BODY_NOT_FOUND", "IK target references an unknown RobotSpec body", field=f"ik_targets[{index}].body_name", actual=body_name, expected=self.robot.body_names))
                offset = np.asarray(target.get("target_offset", []), dtype=np.float64)
                if offset.shape != (3,) or not np.isfinite(offset).all():
                    issues.append(_issue("IK_TARGET_INVALID", "IK target_offset must be a finite 3D vector", field=f"ik_targets[{index}].target_offset", actual=list(offset.shape), expected=[3]))
            if self.ik_solver is None:
                issues.append(_issue("IK_SOLVER_UNAVAILABLE", "IK targets require a registered server-side solver", field="ik_targets"))
            else:
                try:
                    edited = self.ik_solver.apply(edited, config.ik_targets)
                except Exception as exc:  # adapter boundary converts vendor errors into a stable issue
                    issues.append(_issue("IK_FAILED", f"constrained IK failed: {exc}", field="ik_targets"))

        edited = _resample(edited, transform.time_scale)
        if config.filters.get("smooth", False) and len(edited.joint_pos) >= 3:
            edited.joint_pos[1:-1] = (edited.joint_pos[:-2] + edited.joint_pos[1:-1] + edited.joint_pos[2:]) / 3.0
        return self._result(edited if not any(item.severity == ValidationSeverity.BLOCKING_ERROR for item in issues) else None, issues, edited, check_velocity=config.filters.get("max_velocity_check", True))

    def _result(self, arrays: MotionArrays | None, issues: list[ValidationIssue], source: MotionArrays, *, check_velocity: bool = False) -> MotionEditResult:
        candidate = arrays
        if candidate is not None:
            if not np.isfinite(candidate.joint_pos).all() or not np.isfinite(candidate.root_pos).all() or not np.isfinite(candidate.root_rot).all():
                issues.append(_issue("NONFINITE_VALUE", "edited trajectory contains NaN or Inf"))
            limit_violations = 0
            for index, joint in enumerate(self.robot.joints):
                values = candidate.joint_pos[:, index]
                limit_violations += int(np.count_nonzero((values < joint.position_limit[0] - 1e-6) | (values > joint.position_limit[1] + 1e-6)))
            total = max(1, candidate.joint_pos.size)
            limit_ratio = limit_violations / total
            if limit_violations:
                issues.append(_issue("JOINT_LIMIT_VIOLATION", "edited trajectory exceeds a hard G1 joint limit", field="joint_pos", actual=limit_ratio, expected=0.0))
            if np.min(candidate.root_pos[:, 2]) < -1e-6:
                issues.append(_issue("GROUND_PENETRATION", "root trajectory goes below the ground plane", field="root_pos[:,2]", actual=float(np.min(candidate.root_pos[:, 2]),), expected=0.0))
            normalized, invalid = _normalize_quaternions(candidate.root_rot)
            if invalid.any():
                issues.append(_issue("ZERO_QUATERNION", "edited trajectory contains a zero quaternion", field="root_rot"))
            candidate = MotionArrays(candidate.fps, candidate.joint_pos, candidate.root_pos, normalized, candidate.joint_names, candidate.quat_convention, candidate.coord_frame)
            max_velocity = float(np.max(np.abs(np.diff(candidate.joint_pos, axis=0)) * candidate.fps)) if len(candidate.joint_pos) > 1 else 0.0
            max_acceleration = float(np.max(np.abs(np.diff(candidate.joint_pos, n=2, axis=0)) * candidate.fps * candidate.fps)) if len(candidate.joint_pos) > 2 else 0.0
            if check_velocity and max_velocity > self.max_velocity_rad_s:
                issues.append(_issue("VELOCITY_LIMIT_WARNING", "edited trajectory exceeds the configured velocity warning threshold", field="joint_pos", actual=max_velocity, expected=self.max_velocity_rad_s, severity=ValidationSeverity.WARNING))
            if check_velocity and max_acceleration > self.max_velocity_rad_s * 10:
                issues.append(_issue("ACCELERATION_LIMIT_WARNING", "edited trajectory exceeds the configured acceleration warning threshold", field="joint_pos", actual=max_acceleration, expected=self.max_velocity_rad_s * 10, severity=ValidationSeverity.WARNING))
        else:
            limit_ratio = 1.0
            max_velocity = 0.0
            max_acceleration = 0.0
        status = "BLOCKED" if any(item.severity == ValidationSeverity.BLOCKING_ERROR for item in issues) else ("WARNING" if issues else "PASS")
        report = MotionQualityReport(status=status, issues=issues, stats={"frame_count": float(len(candidate.joint_pos) if candidate is not None else len(source.joint_pos)), "joint_limit_violation_ratio": float(limit_ratio), "max_joint_velocity_rad_s": float(max_velocity), "max_joint_acceleration_rad_s2": float(max_acceleration)}, compiler_version=COMPILER_VERSION)
        return MotionEditResult(candidate, ValidationResult(valid=status != "BLOCKED", issues=issues, stage="motion_edit_validate", processor_version=COMPILER_VERSION), report)


class MotionEditVersionStore:
    """Thread-safe in-memory repository used until the P2 database port lands."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._versions: dict[str, MotionEditVersion] = {}
        self._by_source: dict[str, list[str]] = {}

    def create(self, config: MotionEditConfig, *, parent_version_id: str | None = None) -> MotionEditVersion:
        with self._lock:
            parent = self._versions.get(parent_version_id) if parent_version_id else None
            if parent and (parent.source_motion_version_id != config.source_motion_version_id or parent.robot_id != config.robot_id):
                raise ValueError("parent motion edit belongs to a different source or robot")
            source_versions = self._by_source.setdefault(config.source_motion_version_id, [])
            version = len(source_versions) + 1
            version_id = str(uuid.uuid4())
            record = MotionEditVersion(version_id=version_id, source_motion_version_id=config.source_motion_version_id, robot_id=config.robot_id, version=version, config=config, config_sha256=_canonical_hash(config.model_dump(mode="json")), parent_version_id=parent_version_id, created_at=datetime.now(timezone.utc).isoformat())
            self._versions[version_id] = record
            source_versions.append(version_id)
            return record

    def get(self, version_id: str) -> MotionEditVersion | None:
        with self._lock:
            return self._versions.get(version_id)

    def list_for_source(self, source_motion_version_id: str) -> list[MotionEditVersion]:
        with self._lock:
            return [self._versions[item] for item in self._by_source.get(source_motion_version_id, [])]

    def restore(self, current_version_id: str, target_version_id: str) -> MotionEditVersion:
        with self._lock:
            current = self._versions.get(current_version_id)
            target = self._versions.get(target_version_id)
            if current is None or target is None:
                raise ValueError("motion edit version to restore was not found")
            if current.source_motion_version_id != target.source_motion_version_id or current.robot_id != target.robot_id:
                raise ValueError("motion edit versions must belong to the same source and robot")
            return self.create(target.config, parent_version_id=current.version_id)


__all__ = ["MotionArrays", "MotionEditResult", "MotionEditVersionStore", "MotionEditor"]
