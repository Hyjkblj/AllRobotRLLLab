"""Run state machine and transition policy."""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    CREATED = "CREATED"
    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    GVHMR_RUNNING = "GVHMR_RUNNING"
    GVHMR_READY = "GVHMR_READY"
    GMR_RUNNING = "GMR_RUNNING"
    RETARGET_READY = "RETARGET_READY"
    MOTION_COMPILING = "MOTION_COMPILING"
    MOTION_READY = "MOTION_READY"
    MOTION_EDITING = "MOTION_EDITING"
    MOTION_VALIDATING = "MOTION_VALIDATING"
    TRAINING_PREPARING = "TRAINING_PREPARING"
    TRAINING = "TRAINING"
    TRAINING_SUCCEEDED = "TRAINING_SUCCEEDED"
    EXPORTING = "EXPORTING"
    EXPORTED = "EXPORTED"
    SIM2SIM_QUEUED = "SIM2SIM_QUEUED"
    SIM2SIM_RUNNING = "SIM2SIM_RUNNING"
    SIM2SIM_PASSED = "SIM2SIM_PASSED"
    READY_TO_DOWNLOAD = "READY_TO_DOWNLOAD"
    FAILED = "FAILED"
    FAILED_NEEDS_REVIEW = "FAILED_NEEDS_REVIEW"
    CANCELLED = "CANCELLED"


_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.UPLOADING, RunStatus.VALIDATING, RunStatus.CANCELLED}),
    RunStatus.UPLOADING: frozenset({RunStatus.UPLOADED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.UPLOADED: frozenset({RunStatus.VALIDATING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.VALIDATING: frozenset({RunStatus.GVHMR_RUNNING, RunStatus.GMR_RUNNING, RunStatus.MOTION_COMPILING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.GVHMR_RUNNING: frozenset({RunStatus.GVHMR_READY, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.GVHMR_READY: frozenset({RunStatus.GMR_RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.GMR_RUNNING: frozenset({RunStatus.RETARGET_READY, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.RETARGET_READY: frozenset({RunStatus.MOTION_COMPILING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.MOTION_COMPILING: frozenset({RunStatus.MOTION_READY, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.MOTION_READY: frozenset({RunStatus.MOTION_EDITING, RunStatus.TRAINING_PREPARING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.MOTION_EDITING: frozenset({RunStatus.MOTION_VALIDATING, RunStatus.CANCELLED}),
    RunStatus.MOTION_VALIDATING: frozenset({RunStatus.MOTION_READY, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.TRAINING_PREPARING: frozenset({RunStatus.TRAINING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.TRAINING: frozenset({RunStatus.TRAINING_SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.TRAINING_SUCCEEDED: frozenset({RunStatus.EXPORTING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.EXPORTING: frozenset({RunStatus.EXPORTED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.EXPORTED: frozenset({RunStatus.SIM2SIM_QUEUED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.SIM2SIM_QUEUED: frozenset({RunStatus.SIM2SIM_RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.SIM2SIM_RUNNING: frozenset({RunStatus.SIM2SIM_PASSED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.SIM2SIM_PASSED: frozenset({RunStatus.READY_TO_DOWNLOAD, RunStatus.FAILED}),
    RunStatus.READY_TO_DOWNLOAD: frozenset(),
    RunStatus.FAILED: frozenset({RunStatus.FAILED_NEEDS_REVIEW, RunStatus.UPLOADED, RunStatus.TRAINING_PREPARING, RunStatus.CANCELLED}),
    RunStatus.FAILED_NEEDS_REVIEW: frozenset({RunStatus.CANCELLED}),
    RunStatus.CANCELLED: frozenset(),
}


class InvalidTransition(ValueError):
    """Raised when a run status transition is not explicitly allowed."""


def can_transition(current: RunStatus, target: RunStatus) -> bool:
    return target in _TRANSITIONS[current]


def transition(current: RunStatus, target: RunStatus) -> RunStatus:
    if not can_transition(current, target):
        raise InvalidTransition(f"invalid run transition: {current} -> {target}")
    return target


__all__ = ["InvalidTransition", "RunStatus", "can_transition", "transition"]

