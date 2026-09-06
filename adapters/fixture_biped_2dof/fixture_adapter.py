"""Framework-only two-DoF RobotAdapter for architecture tests."""

from __future__ import annotations

from pathlib import Path

from backend.app.domain.contracts import (
    ArrayField,
    RetargetMotion,
    RobotSpec,
    RunManifest,
    TrainMotionNPZ,
    TrainingConfig,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


class FixtureBipedAdapter:
    name = "fixture_biped_2dof"
    gmr_robot = "fixture_biped"

    def __init__(self, *, repository_root: Path | None = None) -> None:
        self.repository_root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
        spec_path = Path(__file__).with_name("robot_spec.json")
        self._spec = RobotSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))

    def get_spec(self) -> RobotSpec:
        return self._spec.model_copy(deep=True)

    def self_check(self) -> ValidationResult:
        return ValidationResult.ok(stage="robot_self_check", processor_version=self._spec.adapter_version)

    def validate_motion(self, motion: RetargetMotion) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if motion.robot_id != self.name:
            issues.append(ValidationIssue(code="ROBOT_ID_MISMATCH", message="motion robot_id does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, field="robot_id", expected=self.name, actual=motion.robot_id))
        if motion.joint_names != self._spec.joint_names:
            issues.append(ValidationIssue(code="MOTION_JOINT_ORDER_MISMATCH", message="motion joint order does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, field="joint_names", expected=self._spec.joint_names, actual=motion.joint_names))
        dof = motion.array_meta.get("dof_pos")
        if dof and dof.shape[-1] != self._spec.dof:
            issues.append(ValidationIssue(code="MOTION_DOF_MISMATCH", message="motion DoF does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, field="array_meta.dof_pos.shape", expected=self._spec.dof, actual=dof.shape[-1]))
        return ValidationResult(valid=not issues, issues=issues, stage="fixture_motion_validate", processor_version=self._spec.adapter_version)

    def compile_motion(self, motion: RetargetMotion, config: TrainingConfig, output_dir: Path) -> TrainMotionNPZ:
        validation = self.validate_motion(motion)
        if not validation.valid:
            raise ValueError(validation.model_dump_json())
        source_hash = motion.source.get("sha256")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError("RetargetMotion.source.sha256 is required")
        output_dir.mkdir(parents=True, exist_ok=True)
        shape = [motion.frame_count, self._spec.dof]
        body_shape = [motion.frame_count, len(self._spec.body_names), 3]
        arrays = {
            "joint_pos": ArrayField(path=str(output_dir / "joint_pos.npy"), shape=shape, dtype="float32"),
            "joint_vel": ArrayField(path=str(output_dir / "joint_vel.npy"), shape=shape, dtype="float32"),
            "body_pos_w": ArrayField(path=str(output_dir / "body_pos_w.npy"), shape=body_shape, dtype="float32"),
            "body_quat_w": ArrayField(path=str(output_dir / "body_quat_w.npy"), shape=[motion.frame_count, len(self._spec.body_names), 4], dtype="float32", convention="wxyz"),
            "body_lin_vel_w": ArrayField(path=str(output_dir / "body_lin_vel_w.npy"), shape=body_shape, dtype="float32"),
            "body_ang_vel_w": ArrayField(path=str(output_dir / "body_ang_vel_w.npy"), shape=body_shape, dtype="float32"),
        }
        return TrainMotionNPZ(robot_id=self.name, fps=motion.fps, frame_count=motion.frame_count, joint_names=list(self._spec.joint_names), body_names=list(self._spec.body_names), arrays=arrays, coord_frame=motion.coord_frame, quat_convention="wxyz", source_motion_hash=source_hash, compiler_version="fixture-motion-compiler.v1")

    def validate_training_manifest(self, manifest: RunManifest) -> ValidationResult:
        if manifest.robot.get("robot_id") != self.name:
            return ValidationResult.failure(ValidationIssue(code="MANIFEST_ROBOT_MISMATCH", message="manifest robot does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, expected=self.name, actual=manifest.robot.get("robot_id")), stage="manifest_validate")
        if not manifest.motion.get("train_motion_sha256"):
            return ValidationResult.failure(ValidationIssue(code="MANIFEST_MOTION_HASH_MISSING", message="manifest must pin TrainMotionNPZ hash", severity=ValidationSeverity.BLOCKING_ERROR, field="motion.train_motion_sha256"), stage="manifest_validate")
        return ValidationResult.ok(stage="manifest_validate", processor_version=self._spec.adapter_version)


__all__ = ["FixtureBipedAdapter"]
