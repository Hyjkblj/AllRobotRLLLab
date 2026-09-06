"""Generic RobotSpec-backed adapter implementation.

Robot packages should provide a spec and asset environment mapping instead of
duplicating the platform contract checks.  Vendor-specific control/runtime
integration remains in the concrete package.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
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


class JsonRobotAdapter:
    """Base adapter for robots described by a versioned ``robot_spec.json``."""

    name: str
    gmr_robot: str | None = None
    asset_env: dict[str, str] = {}
    asset_prefixes: tuple[str, ...] = ()
    # Optional adapter-owned command variable for a generic sim2sim runner.
    # The composition root derives a deterministic name when omitted.
    sim2sim_command_env: str | None = None

    def __init__(self, *, repository_root: Path | None = None, spec_path: Path | None = None) -> None:
        self.repository_root = (repository_root or Path(__file__).resolve().parents[2]).resolve()
        self.spec_path = (spec_path or Path(__file__).with_name("robot_spec.json")).resolve()
        self._spec = RobotSpec.model_validate_json(self.spec_path.read_text(encoding="utf-8"))
        if self._spec.robot_id != self.name:
            raise ValueError(f"adapter name does not match RobotSpec.robot_id: {self.name} != {self._spec.robot_id}")

    def get_spec(self) -> RobotSpec:
        return self._spec.model_copy(deep=True)

    def _asset_path(self, key: str) -> Path:
        value = self._spec.assets.get(key, "")
        env_name = self.asset_env.get(key)
        configured = os.getenv(env_name, "").strip() if env_name else ""
        if configured:
            return Path(configured).expanduser().resolve()
        for prefix in self.asset_prefixes:
            if value.startswith(prefix):
                external_root = os.getenv("MUJOCO_MENAGERIE_PATH", "").strip()
                if external_root:
                    return (Path(external_root).expanduser() / value[len(prefix):]).resolve()
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.repository_root / path).resolve()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def asset_identity(self) -> dict[str, dict[str, str | int | bool | None]]:
        result: dict[str, dict[str, str | int | bool | None]] = {}
        for key in self._spec.assets:
            path = self._asset_path(key)
            item: dict[str, str | int | bool | None] = {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": None,
                "size_bytes": None,
            }
            if path.is_file():
                item["sha256"] = self._sha256(path)
                item["size_bytes"] = path.stat().st_size
            result[key] = item
        return result

    def self_check(self) -> ValidationResult:
        model_key = next((key for key in ("mujoco_xml_uri", "mujoco_xml") if key in self._spec.assets), None)
        if model_key is None:
            return ValidationResult.failure(
                ValidationIssue(
                    code="ROBOT_MODEL_ASSET_MISSING",
                    message="RobotSpec does not declare a MuJoCo model asset",
                    severity=ValidationSeverity.BLOCKING_ERROR,
                    field="assets.mujoco_xml_uri",
                ),
                stage="robot_self_check",
            )
        model_path = self._asset_path(model_key)
        if not model_path.is_file():
            return ValidationResult.failure(
                ValidationIssue(
                    code="ROBOT_ASSET_MISSING",
                    message=f"MuJoCo XML not found: {model_path}",
                    severity=ValidationSeverity.BLOCKING_ERROR,
                    field=f"assets.{model_key}",
                ),
                stage="robot_self_check",
            )
        issues: list[ValidationIssue] = []
        try:
            root = ET.parse(model_path).getroot()
        except (OSError, ET.ParseError) as exc:
            return ValidationResult.failure(
                ValidationIssue(code="ROBOT_MODEL_INVALID", message=f"unable to parse MuJoCo XML: {exc}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"assets.{model_key}"),
                stage="robot_self_check",
            )
        joints = [node for node in root.iter("joint") if node.attrib.get("type", "hinge") in {"hinge", "slide"} and node.attrib.get("name")]
        xml_names = [node.attrib["name"] for node in joints]
        if xml_names != self._spec.joint_names:
            issues.append(ValidationIssue(code="ROBOT_JOINT_ORDER_MISMATCH", message="MuJoCo joint order differs from RobotSpec", severity=ValidationSeverity.BLOCKING_ERROR, field="joint_names", expected=self._spec.joint_names, actual=xml_names))
        actuator_names = [node.attrib.get("name") for node in root.findall("./actuator/motor") if node.attrib.get("name")]
        if actuator_names and actuator_names != self._spec.joint_names:
            issues.append(ValidationIssue(code="ROBOT_ACTUATOR_ORDER_MISMATCH", message="MuJoCo actuator order differs from RobotSpec", severity=ValidationSeverity.BLOCKING_ERROR, field="actuation", expected=self._spec.joint_names, actual=actuator_names))
        body_names = {node.attrib["name"] for node in root.iter("body") if node.attrib.get("name")}
        missing = [name for name in self._spec.body_names if name not in body_names]
        if missing:
            issues.append(ValidationIssue(code="ROBOT_BODY_MISSING", message="RobotSpec references missing MuJoCo bodies", severity=ValidationSeverity.BLOCKING_ERROR, field="body_names", expected=self._spec.body_names, actual=missing))
        has_freejoint = root.find(".//freejoint") is not None
        qpos_base = 7 if has_freejoint else 0
        expected_addresses = list(range(qpos_base, qpos_base + self._spec.dof))
        actual_addresses = [joint.qpos_address for joint in self._spec.joints]
        if actual_addresses != expected_addresses:
            issues.append(ValidationIssue(code="ROBOT_QPOS_ADDRESS_MISMATCH", message="RobotSpec qpos addresses are not contiguous", severity=ValidationSeverity.BLOCKING_ERROR, field="joints.qpos_address", expected=expected_addresses, actual=actual_addresses))
        if issues:
            return ValidationResult(valid=False, issues=issues, stage="robot_self_check", processor_version=self._spec.adapter_version)
        return ValidationResult.ok(stage="robot_self_check", processor_version=self._spec.adapter_version)

    def create_kinematics_compiler(self, *, allow_approximation: bool = False):
        from backend.app.runtime.mujoco_kinematics import MuJoCoKinematicsCompiler

        model_key = next((key for key in ("mujoco_xml_uri", "mujoco_xml") if key in self._spec.assets), None)
        if model_key is None:
            raise ValueError(f"RobotSpec does not declare a MuJoCo model asset: {self.name}")
        return MuJoCoKinematicsCompiler(model_path=self._asset_path(model_key), body_names=self._spec.body_names, joint_names=self._spec.joint_names, allow_approximation=allow_approximation)

    def validate_motion(self, motion: RetargetMotion) -> ValidationResult:
        issues: list[ValidationIssue] = []
        if motion.robot_id != self.name:
            issues.append(ValidationIssue(code="ROBOT_ID_MISMATCH", message="motion robot_id does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, field="robot_id", expected=self.name, actual=motion.robot_id))
        if motion.joint_names != self._spec.joint_names:
            issues.append(ValidationIssue(code="MOTION_JOINT_ORDER_MISMATCH", message="motion joint order does not match RobotSpec", severity=ValidationSeverity.BLOCKING_ERROR, field="joint_names", expected=self._spec.joint_names, actual=motion.joint_names))
        field = motion.array_meta.get("dof_pos")
        if field and field.shape[-1] != self._spec.dof:
            issues.append(ValidationIssue(code="MOTION_DOF_MISMATCH", message="motion DoF does not match RobotSpec", severity=ValidationSeverity.BLOCKING_ERROR, field="array_meta.dof_pos.shape", expected=self._spec.dof, actual=field.shape[-1]))
        if motion.quality.nan_count:
            issues.append(ValidationIssue(code="MOTION_NONFINITE", message="motion contains NaN or Inf values", severity=ValidationSeverity.BLOCKING_ERROR, field="quality.nan_count", expected=0, actual=motion.quality.nan_count))
        return ValidationResult(valid=not issues, issues=issues, stage=f"{self.name}_motion_validate", processor_version=self._spec.adapter_version)

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
        return TrainMotionNPZ(robot_id=self.name, fps=motion.fps, frame_count=motion.frame_count, joint_names=list(self._spec.joint_names), body_names=list(self._spec.body_names), arrays=arrays, coord_frame=motion.coord_frame, quat_convention="wxyz", source_motion_hash=source_hash, compiler_version=f"{self.name}-motion-compiler.v1")

    def validate_training_manifest(self, manifest: RunManifest) -> ValidationResult:
        if manifest.robot.get("robot_id") != self.name:
            return ValidationResult.failure(ValidationIssue(code="MANIFEST_ROBOT_MISMATCH", message="manifest robot does not match adapter", severity=ValidationSeverity.BLOCKING_ERROR, expected=self.name, actual=manifest.robot.get("robot_id")), stage="manifest_validate")
        if not manifest.motion.get("train_motion_sha256"):
            return ValidationResult.failure(ValidationIssue(code="MANIFEST_MOTION_HASH_MISSING", message="manifest must pin TrainMotionNPZ hash", severity=ValidationSeverity.BLOCKING_ERROR, field="motion.train_motion_sha256"), stage="manifest_validate")
        return ValidationResult.ok(stage="manifest_validate", processor_version=self._spec.adapter_version)


__all__ = ["JsonRobotAdapter"]
