"""Versioned domain contracts.

The domain layer deliberately contains no FastAPI, persistence, queue, or
simulation imports.  These models are the stable boundary between API,
workers, and robot adapters.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .state_machine import RunStatus


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SchemaVersion(StrEnum):
    ROBOT_SPEC = "robot_spec.v1"
    SOURCE_MOTION = "source_motion_descriptor.v1"
    RETARGET_MOTION = "retarget_motion.v1"
    TRAIN_MOTION = "train_motion_npz.v1"
    MOTION_EDIT = "motion_edit.v1"
    REWARD_CONFIG = "reward_config.v1"
    TRAINING_CONFIG = "training_config.v1"
    RUN_MANIFEST = "run_manifest.v1"


class LicenseInfo(ContractModel):
    status: str = Field(min_length=1)
    source: str | None = None
    source_uri: str | None = None
    processing_scope: str | None = None


class ValidationSeverity(StrEnum):
    BLOCKING_ERROR = "BLOCKING_ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationIssue(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: ValidationSeverity
    field: str | None = None
    actual: Any | None = None
    expected: Any | None = None
    suggested_action: str | None = None


class ValidationResult(ContractModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    stage: str | None = None
    processor_version: str | None = None

    @model_validator(mode="after")
    def valid_matches_issues(self) -> "ValidationResult":
        has_blocking = any(i.severity == ValidationSeverity.BLOCKING_ERROR for i in self.issues)
        if has_blocking and self.valid:
            raise ValueError("valid cannot be true when a blocking issue exists")
        return self

    @classmethod
    def ok(cls, *, stage: str | None = None, processor_version: str | None = None) -> "ValidationResult":
        return cls(valid=True, stage=stage, processor_version=processor_version)

    @classmethod
    def failure(cls, *issues: ValidationIssue, stage: str | None = None) -> "ValidationResult":
        return cls(valid=False, issues=list(issues), stage=stage)


class ProjectRole(StrEnum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class ProjectStatus(StrEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class Actor(ContractModel):
    user_id: str = Field(min_length=1)
    email: str | None = None


class ProjectMember(ContractModel):
    project_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    role: ProjectRole
    created_at: str = Field(min_length=1)


class ProjectRecord(ContractModel):
    project_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    owner_id: str = Field(min_length=1)
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)


class AttemptRecord(ContractModel):
    attempt_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    number: int = Field(ge=1)
    status: RunStatus = RunStatus.CREATED
    created_at: str = Field(min_length=1)
    started_at: str | None = None
    finished_at: str | None = None
    worker_id: str | None = None
    gpu_uuid: str | None = None
    exit_code: int | None = None
    failure_code: str | None = None
    last_heartbeat_at: str | None = None


class RunRecord(ContractModel):
    run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    status: RunStatus = RunStatus.CREATED
    created_by: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    parent_run_id: str | None = None
    current_attempt_id: str = Field(min_length=1)
    manifest: "RunManifest"


class RunEvent(ContractModel):
    seq: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    event_type: Literal["status", "log", "metric", "system"]
    stage: str | None = None
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(min_length=1)


class AuditEvent(ContractModel):
    event_id: str = Field(min_length=1)
    project_id: str | None = None
    actor_id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    resource_type: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(min_length=1)


class OutboxEvent(ContractModel):
    event_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    key: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(min_length=1)
    published_at: str | None = None


class AssetKind(StrEnum):
    VIDEO = "video"
    MOTION = "motion"
    MODEL = "model"


class AssetStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"


class AssetVersionStatus(StrEnum):
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    READY = "READY"
    REJECTED = "REJECTED"


class AssetRecord(ContractModel):
    asset_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    kind: AssetKind
    display_name: str = Field(min_length=1, max_length=255)
    license: LicenseInfo
    status: AssetStatus = AssetStatus.ACTIVE
    created_by: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class AssetVersion(ContractModel):
    asset_version_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    status: AssetVersionStatus = AssetVersionStatus.UPLOADING
    object_key: str = Field(min_length=1)
    original_filename: str = Field(min_length=1, max_length=255)
    content_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(min_length=1)
    validated_at: str | None = None
    rejection_code: str | None = None


class UploadSession(ContractModel):
    session_id: str = Field(min_length=1)
    asset_version_id: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    upload_url: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)
    multipart: bool = False


class ArtifactRecord(ContractModel):
    artifact_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    content_type: str | None = None
    created_at: str = Field(min_length=1)


class MetricPoint(ContractModel):
    attempt_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    name: str = Field(min_length=1)
    value: float
    timestamp: str = Field(min_length=1)


class TaskSubmission(ContractModel):
    """Stable response returned when a long-running stage is queued."""

    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    operation: Literal["train", "export", "sim2sim"]
    queue: str = Field(min_length=1)
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"] = "QUEUED"
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class CheckpointRecord(ContractModel):
    checkpoint_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    iteration: int = Field(ge=0)
    created_at: str = Field(min_length=1)


class ExportFile(ContractModel):
    path: str = Field(min_length=1)
    format: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ExportMetadata(ContractModel):
    schema_version: Literal["export_metadata.v1"] = "export_metadata.v1"
    policy_input_dim: int = Field(gt=0)
    policy_output_dim: int = Field(gt=0)
    action_scale: float = Field(gt=0)
    onnx_opset: int | None = Field(default=None, ge=1)
    input_name: str | None = None
    output_name: str | None = None
    exporter: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    smoke_passed: bool
    files: list[ExportFile] = Field(min_length=1)


class Sim2SimThresholds(ContractModel):
    schema_version: Literal["sim2sim_policy.v1"] = "sim2sim_policy.v1"
    min_survival_rate: float = Field(default=0.90, ge=0, le=1)
    max_joint_rmse_rad: float = Field(default=0.35, gt=0)
    max_root_position_rmse_m: float = Field(default=0.20, gt=0)
    max_orientation_error_deg: float = Field(default=20.0, gt=0)
    max_saturation_ratio: float = Field(default=0.05, ge=0, le=1)
    max_foot_slip_mps: float = Field(default=0.15, ge=0)
    max_seed_metric_spread: float = Field(default=0.20, ge=0)


class SeedEvaluation(ContractModel):
    seed: int
    status: Literal["PASSED", "FAILED"]
    exit_code: int
    duration_seconds: float = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_code: str | None = None
    video_artifact_id: str | None = None
    command: list[str] = Field(default_factory=list)


class Sim2SimReport(ContractModel):
    schema_version: Literal["sim2sim_report.v1"] = "sim2sim_report.v1"
    run_id: str = Field(min_length=1)
    adapter: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    thresholds: Sim2SimThresholds
    evaluations: list[SeedEvaluation] = Field(min_length=1)
    status: Literal["PASSED", "FAILED", "INPUT_QUALITY_FAILED"]
    hard_failures: list[str] = Field(default_factory=list)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PolicyBundle(ContractModel):
    schema_version: Literal["policy_bundle.v1"] = "policy_bundle.v1"
    bundle_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    observation_dim: int = Field(gt=0)
    action_dim: int = Field(gt=0)
    files: list[ExportFile] = Field(min_length=1)
    export: ExportMetadata
    sim2sim_report: Sim2SimReport | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    status: Literal["EXPORTED", "SIM2SIM_PASSED", "READY_TO_DOWNLOAD"] = "EXPORTED"


class ActuationSpec(ContractModel):
    mode: Literal["position_pd"] = "position_pd"
    control_dt: float = Field(gt=0, le=0.2)
    policy_dt: float = Field(gt=0, le=0.2)
    action_scale: float = Field(gt=0, le=2)
    kp_profile: str = Field(min_length=1)
    kd_profile: str = Field(min_length=1)
    torque_limits: list[float] = Field(min_length=1)


class JointSpec(ContractModel):
    name: str = Field(min_length=1)
    qpos_address: int = Field(ge=0)
    axis: tuple[float, float, float]
    position_limit: tuple[float, float]
    torque_limit: float = Field(gt=0)

    @field_validator("position_limit")
    @classmethod
    def limits_are_ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] >= value[1]:
            raise ValueError("position_limit lower bound must be less than upper bound")
        return value


class RobotSpec(ContractModel):
    schema_version: Literal[SchemaVersion.ROBOT_SPEC] = SchemaVersion.ROBOT_SPEC
    robot_id: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    assets: dict[str, str] = Field(default_factory=dict)
    joint_names: list[str] = Field(min_length=1)
    joints: list[JointSpec] = Field(min_length=1)
    body_names: list[str] = Field(min_length=1)
    dof: int = Field(gt=0)
    actuation: ActuationSpec
    capabilities: list[str] = Field(default_factory=list)
    gmr_mapping_version: str | None = None
    isaac_task_ids: list[str] = Field(default_factory=list)
    sim2sim_adapter: str = Field(min_length=1)
    license: LicenseInfo

    @model_validator(mode="after")
    def check_joint_contract(self) -> "RobotSpec":
        names = [joint.name for joint in self.joints]
        if self.dof != len(self.joint_names) or self.dof != len(self.joints):
            raise ValueError("dof must match joint_names and joints length")
        if names != self.joint_names:
            raise ValueError("joints must use the same ordered names as joint_names")
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")
        if len(self.actuation.torque_limits) != self.dof:
            raise ValueError("actuation torque_limits must cover every DoF")
        return self


class ArrayField(ContractModel):
    path: str = Field(min_length=1)
    shape: list[int] = Field(min_length=1)
    dtype: str = Field(min_length=1)
    convention: str | None = None
    value: Any | None = None


class SourceMotionDescriptor(ContractModel):
    schema_version: Literal[SchemaVersion.SOURCE_MOTION] = SchemaVersion.SOURCE_MOTION
    asset_version_id: str = Field(min_length=1)
    file_format: Literal["pt", "npz", "csv", "pkl"]
    detected_type: Literal["g1_joint_trajectory", "human_pose", "gvhmr_result", "unsupported"]
    source_skeleton: str | None = None
    fields: dict[str, ArrayField] = Field(default_factory=dict)
    joint_names: list[str] = Field(default_factory=list)
    coord_frame: str | None = None
    quaternion_convention: Literal["xyzw", "wxyz"] | None = None
    license: LicenseInfo
    detector_version: str = Field(min_length=1)


class MotionQuality(ContractModel):
    nan_count: int = Field(ge=0)
    quat_norm_max_error: float = Field(ge=0)
    joint_limit_violation_ratio: float = Field(ge=0, le=1)
    foot_sliding_ratio: float = Field(ge=0, le=1)


class MotionArrayMeta(ContractModel):
    dtype: str = Field(min_length=1)
    shape: list[int] = Field(min_length=1)
    convention: str | None = None


class RetargetMotion(ContractModel):
    format_version: Literal[SchemaVersion.RETARGET_MOTION] = SchemaVersion.RETARGET_MOTION
    robot_id: str = Field(min_length=1)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=1)
    arrays: dict[str, str] = Field(min_length=1)
    array_meta: dict[str, MotionArrayMeta] = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    coord_frame: str = Field(min_length=1)
    source: dict[str, str] = Field(min_length=1)
    quality: MotionQuality
    converter: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def check_motion_shapes(self) -> "RetargetMotion":
        dof_pos = self.array_meta.get("dof_pos")
        if dof_pos and dof_pos.shape != [self.frame_count, len(self.joint_names)]:
            raise ValueError("dof_pos shape must match frame_count and joint_names")
        return self


class TrainMotionNPZ(ContractModel):
    format_version: Literal[SchemaVersion.TRAIN_MOTION] = SchemaVersion.TRAIN_MOTION
    robot_id: str = Field(min_length=1)
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=1)
    joint_names: list[str] = Field(min_length=1)
    body_names: list[str] = Field(min_length=1)
    arrays: dict[str, ArrayField] = Field(min_length=1)
    coord_frame: str = Field(min_length=1)
    quat_convention: Literal["xyzw", "wxyz"]
    source_motion_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compiler_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_required_arrays(self) -> "TrainMotionNPZ":
        required = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
        missing = required.difference(self.arrays)
        if missing:
            raise ValueError(f"missing required TrainMotionNPZ arrays: {sorted(missing)}")
        joint_shape = self.arrays["joint_pos"].shape
        if joint_shape != [self.frame_count, len(self.joint_names)]:
            raise ValueError("joint_pos shape must match frame_count and joint_names")
        velocity_shape = self.arrays["joint_vel"].shape
        if velocity_shape != joint_shape:
            raise ValueError("joint_vel shape must match joint_pos shape")
        body_shape = self.arrays["body_pos_w"].shape
        if body_shape != [self.frame_count, len(self.body_names), 3]:
            raise ValueError("body_pos_w shape must match frame_count, body_names and xyz")
        for name in ("body_lin_vel_w", "body_ang_vel_w"):
            if self.arrays[name].shape != body_shape:
                raise ValueError(f"{name} shape must match body_pos_w shape")
        quat_shape = self.arrays["body_quat_w"].shape
        if quat_shape != [self.frame_count, len(self.body_names), 4]:
            raise ValueError("body_quat_w shape must match frame_count, body_names and quaternion dimension")
        return self


class GlobalTransform(ContractModel):
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_offset: float = 0.0
    time_scale: float = Field(default=1.0, gt=0.1, le=10)


class JointOffset(ContractModel):
    joint_name: str = Field(min_length=1)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    position_offset: float

    @model_validator(mode="after")
    def ordered_frames(self) -> "JointOffset":
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be >= frame_start")
        return self


class MotionEditConfig(ContractModel):
    schema_version: Literal[SchemaVersion.MOTION_EDIT] = SchemaVersion.MOTION_EDIT
    source_motion_version_id: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    global_transform: GlobalTransform = Field(default_factory=GlobalTransform)
    joint_offsets: list[JointOffset] = Field(default_factory=list)
    ik_targets: list[dict[str, Any]] = Field(default_factory=list)
    keyframes: list[dict[str, Any]] = Field(default_factory=list)
    filters: dict[str, bool] = Field(default_factory=lambda: {"smooth": True, "max_velocity_check": True})


class MotionEditVersion(ContractModel):
    """Immutable server-side version of a motion edit configuration."""

    version_id: str = Field(min_length=1)
    source_motion_version_id: str = Field(min_length=1)
    robot_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    config: MotionEditConfig
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_version_id: str | None = None
    created_at: str = Field(min_length=1)


class MotionQualityReport(ContractModel):
    status: Literal["PASS", "WARNING", "BLOCKED"]
    issues: list[ValidationIssue] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)
    compiler_version: str = Field(min_length=1)


class RewardConfigVersion(ContractModel):
    """Immutable server-side version of a validated reward configuration."""

    version_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    config: "RewardConfig"
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_version_id: str | None = None
    created_at: str = Field(min_length=1)


class RewardTermSpec(ContractModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    parameter_schema: dict[str, Any] = Field(default_factory=dict)
    default_weight: float
    weight_range: tuple[float, float]
    applicable_robots: list[str] = Field(min_length=1)
    applicable_tasks: list[str] = Field(min_length=1)
    hard_termination: bool = False
    implementation_version: str = Field(min_length=1)

    @field_validator("weight_range")
    @classmethod
    def weight_range_is_ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] > value[1]:
            raise ValueError("weight_range lower bound must not exceed upper bound")
        return value


class RewardTerm(ContractModel):
    id: str = Field(min_length=1)
    enabled: bool = True
    weight: float
    params: dict[str, Any] = Field(default_factory=dict)


class RewardConfig(ContractModel):
    schema_version: Literal[SchemaVersion.REWARD_CONFIG] = SchemaVersion.REWARD_CONFIG
    base_template: str = Field(min_length=1)
    parent_config_id: str | None = None
    terms: list[RewardTerm] = Field(min_length=1)
    terminations: list[str] = Field(min_length=1)
    annealing: list[dict[str, Any]] = Field(default_factory=list)


RewardConfigVersion.model_rebuild()


class ObservationConfig(ContractModel):
    history_length: int = Field(default=3, ge=1, le=32)
    include_root_velocity: bool = True
    include_projected_gravity: bool = True
    include_reference: bool = True
    clip_value: float = Field(default=100.0, gt=0)


class ActionConfig(ContractModel):
    mode: Literal["joint_position_delta"] = "joint_position_delta"
    scale: float = Field(default=0.25, gt=0, le=2)
    clip: float = Field(default=1.0, gt=0, le=10)


class ControlConfig(ContractModel):
    decimation: int = Field(default=1, ge=1, le=16)
    kp_profile: str = "g1_default"
    kd_profile: str = "g1_default"


class PPOConfig(ContractModel):
    algorithm: Literal["rsl_rl_ppo"] = "rsl_rl_ppo"
    seed: int = 1234
    num_envs: int = Field(default=4096, ge=1, le=100000)
    max_iterations: int = Field(default=5000, ge=1, le=1000000)
    rollout_length: int = Field(default=24, ge=1, le=4096)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    gamma: float = Field(default=0.99, gt=0, lt=1)
    lam: float = Field(default=0.95, gt=0, lt=1)
    clip_param: float = Field(default=0.2, gt=0, le=1)
    entropy_coef: float = Field(default=0.01, ge=0, le=1)
    value_loss_coef: float = Field(default=1.0, ge=0, le=10)
    max_grad_norm: float = Field(default=1.0, gt=0, le=100)
    hidden_dims: list[int] = Field(default_factory=lambda: [512, 256, 128], min_length=1)


class DomainRandomization(ContractModel):
    enabled: bool = True
    mass_scale: tuple[float, float] = (0.95, 1.05)
    friction: tuple[float, float] = (0.7, 1.3)
    motor_strength: tuple[float, float] = (0.95, 1.05)


class ResourceRequest(ContractModel):
    gpu_count: int = Field(default=1, ge=1, le=2)
    gpu_memory_gb: float = Field(default=8, gt=0)
    cpu_cores: int = Field(default=8, ge=1)
    shared_memory_gb: float = Field(default=8, gt=0)
    exclusive_gpu: bool = False


class TrainingConfig(ContractModel):
    schema_version: Literal[SchemaVersion.TRAINING_CONFIG] = SchemaVersion.TRAINING_CONFIG
    task_id: Literal["g1_mimic"] = "g1_mimic"
    scene_id: str = "g1_flat"
    motion_asset_version_id: str = Field(min_length=1)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    domain_randomization: DomainRandomization = Field(default_factory=DomainRandomization)
    resources: ResourceRequest = Field(default_factory=ResourceRequest)


class RuntimeVersions(ContractModel):
    isaac_lab_git: str = "v2.3.0@3c6e67bb5"
    isaac_lab_package: str = "0.47.2"
    isaac_sim_package: str = "5.1.0.0"
    unitree_rl_lab_package: str = "0.2.1"
    unitree_mujoco_git: str = "ae6a840"
    gmr_git: str = "bb1bbe4"
    gvhmr_git: str = "6ec3ca3"
    mujoco_runtime: str = "pending"
    python: str = "3.11"
    torch: str = "pending"
    cuda_driver: str = "pending"
    container_digest: str = "pending"


class RunManifest(ContractModel):
    schema_version: Literal[SchemaVersion.RUN_MANIFEST] = SchemaVersion.RUN_MANIFEST
    project_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    parent_run_id: str | None = None
    robot: dict[str, str] = Field(min_length=1)
    motion: dict[str, Any] = Field(min_length=1)
    reward_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: RuntimeVersions = Field(default_factory=RuntimeVersions)
    execution: dict[str, Any] = Field(default_factory=dict)
    licenses: list[LicenseInfo] = Field(default_factory=list)
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    def canonical_bytes(self, *, include_hash: bool = False) -> bytes:
        data = self.model_dump(mode="json")
        if not include_hash:
            data.pop("manifest_sha256", None)
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def computed_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def freeze(self) -> "RunManifest":
        digest = self.computed_hash()
        if self.manifest_sha256 and self.manifest_sha256 != digest:
            raise ValueError("manifest_sha256 does not match canonical manifest")
        return self.model_copy(update={"manifest_sha256": digest})


RunRecord.model_rebuild()


__all__ = [
    "Actor", "ActuationSpec", "ActionConfig", "ArrayField", "ArtifactRecord", "AssetKind", "AssetRecord", "AssetStatus", "AssetVersion", "AssetVersionStatus", "AttemptRecord", "AuditEvent", "CheckpointRecord", "ControlConfig", "DomainRandomization", "ExportFile", "ExportMetadata",
    "GlobalTransform", "JointOffset", "JointSpec", "LicenseInfo", "MotionArrayMeta",
    "MetricPoint", "MotionEditConfig", "MotionEditVersion", "MotionQuality", "MotionQualityReport", "OutboxEvent", "PPOConfig", "PolicyBundle", "ProjectMember", "ProjectRecord", "ProjectRole", "ProjectStatus", "ResourceRequest", "RetargetMotion", "TaskSubmission",
    "RewardConfig", "RewardConfigVersion", "RewardTerm", "RewardTermSpec", "RobotSpec", "RunEvent", "RunManifest", "RunRecord", "RunStatus", "RuntimeVersions", "SeedEvaluation", "Sim2SimReport", "Sim2SimThresholds", "UploadSession",
    "SchemaVersion", "SourceMotionDescriptor", "TrainMotionNPZ", "TrainingConfig", "ValidationIssue",
    "ValidationResult", "ValidationSeverity",
]
