"""Server-owned reward term registry and configuration validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import nullcontext

from backend.app.domain.contracts import (
    RewardConfig,
    RewardConfigVersion,
    RewardTermSpec,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from backend.app.infrastructure.local_file import FileLock


SAFETY_TERMINATIONS = frozenset({"timeout", "bad_anchor_orientation", "fall", "joint_limit", "nan_inf"})


class RewardRegistry:
    """Task/robot-scoped registry for server-owned reward terms."""

    def __init__(self, terms: list[RewardTermSpec] | None = None) -> None:
        self._terms = tuple(terms or default_reward_catalog())

    def list(self, *, robot_id: str | None = None, task_id: str | None = None) -> list[RewardTermSpec]:
        return [term for term in self._terms if (robot_id is None or robot_id in term.applicable_robots or "*" in term.applicable_robots) and (task_id is None or task_id in term.applicable_tasks or "*" in term.applicable_tasks)]

    def validate(self, config: RewardConfig, *, robot_id: str, task_id: str) -> ValidationResult:
        return validate_reward_config(config, robot_id=robot_id, task_id=task_id, catalog=self._terms)


def default_reward_catalog() -> list[RewardTermSpec]:
    return [
        RewardTermSpec(id="tracking.joint_pos", description="reference joint position tracking", unit="rad", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 10, "default": 0.25}}, default_weight=1.0, weight_range=(-10.0, 10.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="tracking.joint_pos.v1"),
        RewardTermSpec(id="tracking.joint_vel", description="reference joint velocity tracking", unit="rad/s", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 20, "default": 0.5}}, default_weight=0.2, weight_range=(-10.0, 10.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="tracking.joint_vel.v1"),
        RewardTermSpec(id="tracking.root_pose", description="reference root pose tracking", unit="m/rad", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 10, "default": 0.2}}, default_weight=0.5, weight_range=(-10.0, 10.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="tracking.root_pose.v1"),
        RewardTermSpec(id="tracking.body_pose", description="reference body pose tracking", unit="m/rad", parameter_schema={"sigma": {"type": "number", "minimum": 0.001, "maximum": 10, "default": 0.2}}, default_weight=0.5, weight_range=(-10.0, 10.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="tracking.body_pose.v1"),
        RewardTermSpec(id="regularization.action_rate", description="penalize action changes", unit="1", parameter_schema={}, default_weight=-0.02, weight_range=(-10.0, 0.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="regularization.action_rate.v1"),
        RewardTermSpec(id="regularization.torque", description="penalize actuator effort", unit="Nm", parameter_schema={}, default_weight=-0.001, weight_range=(-10.0, 0.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="regularization.torque.v1"),
        RewardTermSpec(id="stability.contact", description="maintain expected contacts", unit="1", parameter_schema={}, default_weight=0.1, weight_range=(-10.0, 10.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="stability.contact.v1"),
        RewardTermSpec(id="stability.foot_slip", description="penalize foot sliding", unit="m/s", parameter_schema={}, default_weight=-0.1, weight_range=(-10.0, 0.0), applicable_robots=["*"], applicable_tasks=["*"], implementation_version="stability.foot_slip.v1"),
    ]


def default_reward_config(*, robot_id: str | None = None, task_id: str | None = None, registry: RewardRegistry | None = None) -> RewardConfig:
    """Build a scoped reward config without assuming a particular robot.

    Production callers should pass both identifiers. The unscoped form
    produces a generic template that can be validated against any wildcard
    registry entry; it never silently selects G1.
    """

    scope_robot = robot_id or "*"
    scope_task = task_id or "*"
    catalog = (registry or RewardRegistry()).list(robot_id=scope_robot, task_id=scope_task)
    preferred_ids = ("tracking.joint_pos", "tracking.joint_vel", "tracking.root_pose", "regularization.action_rate", "regularization.torque")
    by_id = {term.id: term for term in catalog}
    selected = [by_id[item] for item in preferred_ids if item in by_id]
    selected.extend(term for term in catalog if term.id not in preferred_ids)
    if not selected:
        raise ValueError(f"no reward terms are registered for {scope_robot}/{scope_task}")
    terms = []
    for term in selected:
        params = {name: schema["default"] for name, schema in term.parameter_schema.items() if "default" in schema}
        terms.append({"id": term.id, "enabled": True, "weight": term.default_weight, "params": params})
    return RewardConfig(
        base_template=f"{scope_task}_v1" if task_id else "mimic_v1",
        terms=terms,
        terminations=sorted(SAFETY_TERMINATIONS),
    )


def validate_reward_config(config: RewardConfig, *, robot_id: str | None = None, task_id: str | None = None, catalog: list[RewardTermSpec] | tuple[RewardTermSpec, ...] | None = None) -> ValidationResult:
    robot_scope = robot_id or "*"
    task_scope = task_id or "*"
    registry = {item.id: item for item in (catalog or default_reward_catalog())}
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
        robot_applies = robot_scope in spec.applicable_robots or "*" in spec.applicable_robots
        task_applies = task_scope in spec.applicable_tasks or "*" in spec.applicable_tasks
        if not robot_applies or not task_applies:
            issues.append(ValidationIssue(code="REWARD_TERM_NOT_APPLICABLE", message=f"reward term does not apply to {robot_scope}/{task_scope}: {term.id}", severity=ValidationSeverity.BLOCKING_ERROR, field=f"terms.{term.id}"))
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
    """Immutable reward version repository with optional JSON persistence."""

    def __init__(self, registry: RewardRegistry | None = None, *, storage_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._versions: dict[str, RewardConfigVersion] = {}
        self._by_template: dict[str, list[str]] = {}
        self._registry = registry or RewardRegistry()
        self.storage_path = Path(storage_path).expanduser().resolve() if storage_path else None
        self._file_lock = FileLock(self.storage_path.with_suffix(self.storage_path.suffix + ".lock")) if self.storage_path else None
        if self.storage_path is not None:
            self._load()

    def create(self, config: RewardConfig, *, parent_version_id: str | None = None, robot_id: str | None = None, task_id: str | None = None, registry: RewardRegistry | None = None) -> RewardConfigVersion:
        validation = (registry or self._registry).validate(config, robot_id=robot_id, task_id=task_id)
        if not validation.valid:
            raise ValueError(validation.model_dump_json())
        with self._lock:
            with (self._file_lock or nullcontext()):
                if self.storage_path is not None:
                    self._load()
                parent = self._versions.get(parent_version_id) if parent_version_id else None
                if parent and parent.config.base_template != config.base_template:
                    raise ValueError("parent reward config belongs to a different template")
                versions = self._by_template.setdefault(config.base_template, [])
                version_id = str(uuid.uuid4())
                canonical = json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                record = RewardConfigVersion(version_id=version_id, version=len(versions) + 1, config=config, config_sha256=hashlib.sha256(canonical).hexdigest(), parent_version_id=parent_version_id, created_at=datetime.now(timezone.utc).isoformat())
                self._versions[version_id] = record
                versions.append(version_id)
                self._persist()
                return record

    def get(self, version_id: str) -> RewardConfigVersion | None:
        with self._lock:
            self._refresh()
            return self._versions.get(version_id)

    def list_for_template(self, template: str) -> list[RewardConfigVersion]:
        with self._lock:
            self._refresh()
            return [self._versions[item] for item in self._by_template.get(template, [])]

    def get_by_sha256(self, config_sha256: str) -> RewardConfigVersion | None:
        with self._lock:
            self._refresh()
            return next((record for record in self._versions.values() if record.config_sha256 == config_sha256), None)

    def validate(self, config: RewardConfig, *, robot_id: str, task_id: str) -> ValidationResult:
        return self._registry.validate(config, robot_id=robot_id, task_id=task_id)

    def _load(self) -> None:
        assert self.storage_path is not None
        self._versions.clear()
        self._by_template.clear()
        if not self.storage_path.is_file():
            return
        try:
            values = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if not isinstance(values, list):
                raise ValueError("reward config store must contain a JSON list")
            for value in values:
                record = RewardConfigVersion.model_validate(value)
                self._versions[record.version_id] = record
                self._by_template.setdefault(record.config.base_template, []).append(record.version_id)
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f"unable to load reward config store: {self.storage_path}") from exc

    def _refresh(self) -> None:
        if self.storage_path is None:
            return
        with (self._file_lock or nullcontext()):
            self._load()

    def _persist(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([record.model_dump(mode="json") for record in self._versions.values()], ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=".reward-configs.", suffix=".tmp", dir=self.storage_path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.storage_path)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["RewardRegistry", "SAFETY_TERMINATIONS", "RewardConfigVersionStore", "default_reward_catalog", "default_reward_config", "validate_reward_config"]
