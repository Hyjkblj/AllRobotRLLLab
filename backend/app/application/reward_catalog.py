"""Server-owned reward term registry and configuration validation."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone

from backend.app.domain.contracts import (
    RewardConfig,
    RewardConfigVersion,
    RewardTermSpec,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)


SAFETY_TERMINATIONS = frozenset({"timeout", "bad_anchor_orientation", "fall", "joint_limit", "nan_inf"})


def default_reward_catalog() -> list[RewardTermSpec]:
    return [
        RewardTermSpec(id="tracking.joint_pos", description="reference joint position tracking", unit="rad", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 10, "default": 0.25}}, default_weight=1.0, weight_range=(-10.0, 10.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="tracking.joint_pos.v1"),
        RewardTermSpec(id="tracking.joint_vel", description="reference joint velocity tracking", unit="rad/s", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 20, "default": 0.5}}, default_weight=0.2, weight_range=(-10.0, 10.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="tracking.joint_vel.v1"),
        RewardTermSpec(id="tracking.root_pose", description="reference root pose tracking", unit="m/rad", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 10, "default": 0.2}}, default_weight=0.5, weight_range=(-10.0, 10.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="tracking.root_pose.v1"),
        RewardTermSpec(id="tracking.body_pose", description="reference body pose tracking", unit="m/rad", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 10, "default": 0.2}}, default_weight=0.5, weight_range=(-10.0, 10.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="tracking.body_pose.v1"),
        RewardTermSpec(id="regularization.action_rate", description="penalize action changes", unit="1", parameter_schema={}, default_weight=-0.02, weight_range=(-10.0, 0.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="regularization.action_rate.v1"),
        RewardTermSpec(id="regularization.torque", description="penalize actuator effort", unit="Nm", parameter_schema={}, default_weight=-0.001, weight_range=(-10.0, 0.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="regularization.torque.v1"),
        RewardTermSpec(id="stability.contact", description="maintain expected contacts", unit="1", parameter_schema={}, default_weight=0.1, weight_range=(-10.0, 10.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="stability.contact.v1"),
        RewardTermSpec(id="stability.foot_slip", description="penalize foot sliding", unit="m/s", parameter_schema={}, default_weight=-0.1, weight_range=(-10.0, 0.0), applicable_robots=["unitree_g1_29dof"], applicable_tasks=["g1_mimic"], implementation_version="stability.foot_slip.v1"),
    ]


def default_reward_config() -> RewardConfig:
    return RewardConfig(
        base_template="g1_mimic_v1",
        terms=[
            {"id": "tracking.joint_pos", "enabled": True, "weight": 1.0, "params": {"sigma": 0.25}},
            {"id": "tracking.joint_vel", "enabled": True, "weight": 0.2, "params": {"sigma": 0.5}},
            {"id": "tracking.root_pose", "enabled": True, "weight": 0.5, "params": {"sigma": 0.2}},
            {"id": "regularization.action_rate", "enabled": True, "weight": -0.02, "params": {}},
            {"id": "regularization.torque", "enabled": True, "weight": -0.001, "params": {}},
        ],
        terminations=sorted(SAFETY_TERMINATIONS),
    )


def validate_reward_config(config: RewardConfig, *, robot_id: str = "unitree_g1_29dof", task_id: str = "g1_mimic") -> ValidationResult:
    registry = {item.id: item for item in default_reward_catalog()}
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    for term in config.terms:
        if term.id in seen:
            issues.append(ValidationIssue(code="REWARD_TERM_DUPLICATE", message=f"duplicate reward term: {term.id}", severity=ValidationSeverity.BLOCKING_ERROR, field="terms"))
            continue
        seen.add(term.id)
        spec = registry.get(term.id)
        if spec is None:
            issues.append(ValidationIssue(code="REWARD_TERM_NOT_REGISTERED", message=f"reward term is not registered: {term.id}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}"))
            continue
        if robot_id not in spec.applicable_robots or task_id not in spec.applicable_tasks:
            issues.append(ValidationIssue(code="REWARD_TERM_NOT_APPLICABLE", message=f"reward term does not apply to {robot_id}/{task_id}: {term.id}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}"))
        if not spec.weight_range[0] <= term.weight <= spec.weight_range[1]:
            issues.append(ValidationIssue(code="REWARD_WEIGHT_OUT_OF_RANGE", message=f"reward weight out of range: {term.id}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}.weight", expected=spec.weight_range, actual=term.weight))
        allowed_params = set(spec.parameter_schema)
        unknown_params = set(term.params).difference(allowed_params)
        if unknown_params:
            issues.append(ValidationIssue(code="REWARD_PARAM_NOT_REGISTERED", message=f"reward parameters are not registered: {sorted(unknown_params)}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}.params", expected=sorted(allowed_params), actual=sorted(term.params)))
        for param_name, value in term.params.items():
            schema = spec.parameter_schema.get(param_name)
            if not schema:
                continue
            value_type = schema.get("type")
            type_ok = {"number": isinstance(value, (int, float)) and not isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool), "boolean": isinstance(value, bool), "string": isinstance(value, str)}.get(value_type, True)
            if not type_ok:
                issues.append(ValidationIssue(code="REWARD_PARAM_TYPE_INVALID", message=f"reward parameter has the wrong type: {term.id}.{param_name}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}.params.{param_name}", expected=value_type, actual=type(value).__name__))
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
                    issues.append(ValidationIssue(code="REWARD_PARAM_OUT_OF_RANGE", message=f"reward parameter is outside its registered range: {term.id}.{param_name}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}.params.{param_name}", expected={key: schema[key] for key in ("minimum", "maximum") if key in schema}, actual=value))
    missing_safety = SAFETY_TERMINATIONS.difference(config.terminations)
    if missing_safety:
        issues.append(ValidationIssue(code="SAFETY_TERMINATION_REQUIRED", message="safety terminations cannot be disabled", severity=ValidationSeverity.BLOCKING_ERROR, field="terminations", expected=sorted(SAFETY_TERMINATIONS), actual=config.terminations))
    return ValidationResult(valid=not issues, issues=issues, stage="reward_config_validate", processor_version="reward-registry.v1")


class RewardConfigVersionStore:
    """Immutable in-memory version repository for the P1 Reward Builder."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._versions: dict[str, RewardConfigVersion] = {}
        self._by_template: dict[str, list[str]] = {}

    def create(self, config: RewardConfig, *, parent_version_id: str | None = None) -> RewardConfigVersion:
        validation = validate_reward_config(config)
        if not validation.valid:
            raise ValueError(validation.model_dump_json())
        with self._lock:
            parent = self._versions.get(parent_version_id) if parent_version_id else None
            if parent and parent.config.base_template != config.base_template:
                raise ValueError("parent reward config belongs to a different template")
            versions = self._by_template.setdefault(config.base_template, [])
            version_id = str(uuid.uuid4())
            canonical = json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            record = RewardConfigVersion(version_id=version_id, version=len(versions) + 1, config=config, config_sha256=hashlib.sha256(canonical).hexdigest(), parent_version_id=parent_version_id, created_at=datetime.now(timezone.utc).isoformat())
            self._versions[version_id] = record
            versions.append(version_id)
            return record

    def get(self, version_id: str) -> RewardConfigVersion | None:
        with self._lock:
            return self._versions.get(version_id)

    def list_for_template(self, template: str) -> list[RewardConfigVersion]:
        with self._lock:
            return [self._versions[item] for item in self._by_template.get(template, [])]


__all__ = ["SAFETY_TERMINATIONS", "RewardConfigVersionStore", "default_reward_catalog", "default_reward_config", "validate_reward_config"]
