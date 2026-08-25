"""Unitree G1 adapter implementation.

Only G1-specific asset and contract checks live here.  The adapter does not
know about HTTP, persistence, Celery, or Isaac imports.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.app.domain.contracts import (
    RetargetMotion,
    RobotSpec,
    RunManifest,
    TrainMotionNPZ,
    TrainingConfig,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


class UnitreeG1Adapter:
    """Validate the repository G1 29 DoF assets against the published spec."""

    name = "unitree_g1_29dof"

    def __init__(self, *, repository_root: Path | None = None, spec_path: Path | None = None) -> None:
        root = repository_root or Path(__file__).resolve().parents[2]
        self.repository_root = root.resolve()
        self.spec_path = (spec_path or Path(__file__).with_name("robot_spec.json")).resolve()
        self._spec = RobotSpec.model_validate_json(self.spec_path.read_text(encoding="utf-8"))

    def get_spec(self) -> RobotSpec:
        return self._spec.model_copy(deep=True)

    def create_ik_solver(self):
        """Create a lazy G1 IK adapter without importing MuJoCo at API startup."""

        from .ik_solver import G1MuJoCoIKSolver

        return G1MuJoCoIKSolver(repository_root=self.repository_root)

    def _asset_path(self, key: str) -> Path:
        value = self._spec.assets[key]
        return (self.repository_root / value).resolve()

    def self_check(self) -> ValidationResult:
        issues: list[ValidationIssue] = []
        xml_path = self._asset_path("mujoco_xml_uri")
        if not xml_path.is_file():
            return ValidationResult.failure(
                ValidationIssue(
                    code="ROBOT_ASSET_MISSING",
                    message=f"MuJoCo XML not found: {xml_path}",
                    severity=ValidationSeverity.BLOCKING_ERROR,
                    field="assets.mujoco_xml_uri",
                    suggested_action="restore the locked G1 adapter asset",
                ),
                stage="robot_self_check",
            )

        root = ET.parse(xml_path).getroot()
        # The MJCF default section also contains an unnamed joint template;
        # only named runtime joints participate in the adapter contract.
        xml_joints = [element for element in root.iter("joint") if element.attrib.get("type", "hinge") == "hinge" and "name" in element.attrib]
        xml_names = [element.attrib["name"] for element in xml_joints]
        if xml_names != self._spec.joint_names:
            issues.append(ValidationIssue(
                code="ROBOT_JOINT_ORDER_MISMATCH",
                message="MuJoCo hinge joint order differs from RobotSpec",
                severity=ValidationSeverity.BLOCKING_ERROR,
                field="joint_names",
                expected=self._spec.joint_names,
                actual=xml_names,
                suggested_action="update the adapter mapping only after verifying both model and actuator order",
            ))

        actuator_names = [element.attrib.get("name") for element in root.findall("./actuator/motor")]
        if actuator_names != self._spec.joint_names:
            issues.append(ValidationIssue(
                code="ROBOT_ACTUATOR_ORDER_MISMATCH",
                message="MuJoCo actuator order differs from RobotSpec",
                severity=ValidationSeverity.BLOCKING_ERROR,
                field="actuation",
                expected=self._spec.joint_names,
                actual=actuator_names,
            ))

        xml_body_names = {element.attrib["name"] for element in root.iter("body") if "name" in element.attrib}
        missing_bodies = [name for name in self._spec.body_names if name not in xml_body_names]
        if missing_bodies:
            issues.append(ValidationIssue(
                code="ROBOT_BODY_MISSING",
                message="RobotSpec references body names absent from MuJoCo XML",
                severity=ValidationSeverity.BLOCKING_ERROR,
                field="body_names",
                expected=self._spec.body_names,
                actual=sorted(xml_body_names),
                suggested_action="update the adapter body mapping against the locked model",
            ))

        has_freejoint = root.find(".//freejoint") is not None
        qpos_base = 7 if has_freejoint else 1
        actual_qpos_addresses = [joint.qpos_address for joint in self._spec.joints]
        expected_qpos_addresses = list(range(qpos_base, qpos_base + self._spec.dof))
        if actual_qpos_addresses != expected_qpos_addresses:
            issues.append(ValidationIssue(
                code="ROBOT_QPOS_ADDRESS_MISMATCH",
                message="RobotSpec qpos addresses are not contiguous after the root joint",
                severity=ValidationSeverity.BLOCKING_ERROR,
                field="joints.qpos_address",
                expected=expected_qpos_addresses,
                actual=actual_qpos_addresses,
            ))

        for expected, element in zip(self._spec.joints, xml_joints):
            actual_range = tuple(float(part) for part in element.attrib.get("range", "").split())
            if len(actual_range) == 2 and any(not math.isclose(a, b, abs_tol=1e-4) for a, b in zip(actual_range, expected.position_limit)):
                issues.append(ValidationIssue(
                    code="ROBOT_LIMIT_MISMATCH",
                    message=f"joint limit differs for {expected.name}",
                    severity=ValidationSeverity.BLOCKING_ERROR,
                    field=f"joints.{expected.name}.position_limit",
                    expected=expected.position_limit,
                    actual=actual_range,
                ))

        if issues:
            return ValidationResult(valid=False, issues=issues, stage="robot_self_check")
        return ValidationResult.ok(stage="robot_self_check", processor_version="unitree_g1_adapter.v1")

    def validate_motion(self, motion: RetargetMotion) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if motion.robot_id != self.name:
            issues.append(ValidationIssue(code="ROBOT_ID_MISMATCH", message="motion robot_id does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, field="robot_id", expected=self.name, actual=motion.robot_id))
        if motion.joint_names != self._spec.joint_names:
            issues.append(ValidationIssue(code="MOTION_JOINT_ORDER_MISMATCH", message="motion joint order does not match G1", severity=ValidationSeverity.BLOCKING_ERROR, field="joint_names", expected=self._spec.joint_names, actual=motion.joint_names))
        dof_meta = motion.array_meta.get("dof_pos")
        if dof_meta and dof_meta.shape[-1] != self._spec.dof:
            issues.append(ValidationIssue(code="MOTION_DOF_MISMATCH", message="dof_pos does not contain 29 joints", severity=ValidationSeverity.BLOCKING_ERROR, field="array_meta.dof_pos.shape", expected=[motion.frame_count, self._spec.dof], actual=dof_meta.shape))
        if motion.quality.nan_count:
            issues.append(ValidationIssue(code="MOTION_NONFINITE", message="motion contains NaN or Inf values", severity=ValidationSeverity.BLOCKING_ERROR, field="quality.nan_count", actual=motion.quality.nan_count, expected=0))
        return ValidationResult(valid=not issues, issues=issues, stage="g1_motion_validate")

    def compile_motion(self, motion: RetargetMotion, config: TrainingConfig, output_dir: Path) -> TrainMotionNPZ:
        """Return the output contract; numerical conversion is owned by the worker.

        The worker writes arrays and calls this contract constructor only after
        its NumPy/MuJoCo validation.  Keeping this method metadata-only makes
        API and CPU contract tests independent of Isaac and GPU runtimes.
        """

        result = self.validate_motion(motion)
        if not result.valid:
            raise ValueError(result.model_dump_json())
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "joint_pos": {"path": str(output_dir / "joint_pos.npy"), "shape": [motion.frame_count, self._spec.dof], "dtype": "float32"},
            "joint_vel": {"path": str(output_dir / "joint_vel.npy"), "shape": [motion.frame_count, self._spec.dof], "dtype": "float32"},
            "body_pos_w": {"path": str(output_dir / "body_pos_w.npy"), "shape": [motion.frame_count, len(self._spec.body_names), 3], "dtype": "float32"},
            "body_quat_w": {"path": str(output_dir / "body_quat_w.npy"), "shape": [motion.frame_count, len(self._spec.body_names), 4], "dtype": "float32", "convention": "wxyz"},
            "body_lin_vel_w": {"path": str(output_dir / "body_lin_vel_w.npy"), "shape": [motion.frame_count, len(self._spec.body_names), 3], "dtype": "float32"},
            "body_ang_vel_w": {"path": str(output_dir / "body_ang_vel_w.npy"), "shape": [motion.frame_count, len(self._spec.body_names), 3], "dtype": "float32"},
        }
        source_hash = motion.source.get("sha256")
        if not source_hash or len(source_hash) != 64 or any(character not in "0123456789abcdef" for character in source_hash):
            raise ValueError("RetargetMotion.source.sha256 is required before compiling a TrainMotionNPZ")
        return TrainMotionNPZ(
            robot_id=self.name,
            fps=motion.fps,
            frame_count=motion.frame_count,
            joint_names=self._spec.joint_names,
            body_names=self._spec.body_names,
            arrays=arrays,
            coord_frame=motion.coord_frame,
            quat_convention="wxyz",
            source_motion_hash=source_hash,
            compiler_version="g1-motion-compiler.v1",
        )

    def validate_training_manifest(self, manifest: RunManifest) -> ValidationResult:
        if manifest.robot.get("robot_id") != self.name:
            return ValidationResult.failure(ValidationIssue(code="MANIFEST_ROBOT_MISMATCH", message="manifest robot does not match G1 adapter", severity=ValidationSeverity.BLOCKING_ERROR, expected=self.name, actual=manifest.robot.get("robot_id")), stage="manifest_validate")
        if manifest.motion.get("train_motion_sha256") in (None, ""):
            return ValidationResult.failure(ValidationIssue(code="MANIFEST_MOTION_HASH_MISSING", message="manifest must pin TrainMotionNPZ hash", severity=ValidationSeverity.BLOCKING_ERROR, field="motion.train_motion_sha256"), stage="manifest_validate")
        return ValidationResult.ok(stage="manifest_validate", processor_version="unitree_g1_adapter.v1")


__all__ = ["UnitreeG1Adapter"]
