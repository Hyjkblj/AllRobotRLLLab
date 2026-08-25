"""Cross-contract training configuration validation."""

from __future__ import annotations

from backend.app.domain.contracts import RobotSpec, TrainingConfig, ValidationIssue, ValidationResult, ValidationSeverity


def validate_training_config(config: TrainingConfig, robot: RobotSpec) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if config.task_id not in robot.isaac_task_ids:
        issues.append(ValidationIssue(code="TASK_NOT_REGISTERED", message=f"task is not registered by the selected robot adapter: {config.task_id}", severity=ValidationSeverity.BLOCKING_ERROR, field="task_id", expected=robot.isaac_task_ids, actual=config.task_id))
    if config.action.mode != "joint_position_delta":
        issues.append(ValidationIssue(code="ACTION_MODE_UNSUPPORTED", message="G1 P1 only supports joint_position_delta", severity=ValidationSeverity.BLOCKING_ERROR, field="action.mode", expected="joint_position_delta", actual=config.action.mode))
    if config.action.scale > robot.actuation.action_scale:
        issues.append(ValidationIssue(code="ACTION_SCALE_EXCEEDS_ADAPTER", message="action scale exceeds the adapter safety scale", severity=ValidationSeverity.BLOCKING_ERROR, field="action.scale", expected=robot.actuation.action_scale, actual=config.action.scale))
    expected_decimation = robot.actuation.policy_dt / robot.actuation.control_dt
    if abs(expected_decimation - round(expected_decimation)) > 1e-6:
        issues.append(ValidationIssue(code="CONTROL_DT_INCOMPATIBLE", message="adapter policy_dt must be an integer multiple of control_dt", severity=ValidationSeverity.BLOCKING_ERROR, field="robot.actuation"))
    if config.control.decimation > 1 and config.control.decimation > 16:
        issues.append(ValidationIssue(code="DECIMATION_OUT_OF_RANGE", message="control decimation exceeds the platform bound", severity=ValidationSeverity.BLOCKING_ERROR, field="control.decimation", expected=[1, 16], actual=config.control.decimation))
    if config.resources.gpu_count > 2:
        issues.append(ValidationIssue(code="GPU_COUNT_UNSUPPORTED", message="P1 supports at most two independent GPU resources", severity=ValidationSeverity.BLOCKING_ERROR, field="resources.gpu_count", expected=2, actual=config.resources.gpu_count))
    return ValidationResult(valid=not issues, issues=issues, stage="training_config_validate", processor_version="training-config.v1")


def resume_is_compatible(previous: TrainingConfig, current: TrainingConfig) -> bool:
    """Only experiment fields may change when resuming a checkpoint."""

    structural_fields = ("task_id", "scene_id", "motion_asset_version_id", "observation", "action", "control")
    return all(getattr(previous, field) == getattr(current, field) for field in structural_fields)


__all__ = ["resume_is_compatible", "validate_training_config"]

