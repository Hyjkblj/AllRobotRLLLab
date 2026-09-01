"""Contracts shared by external-process runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RunnerError(RuntimeError):
    """Stable error surfaced when an external runtime cannot complete."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class RuntimeUnavailable(RunnerError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__("RUNTIME_UNAVAILABLE", message, details=details)


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    configured: bool
    available: bool
    path: str | None = None
    revision: str | None = None
    expected_revision: str | None = None
    python: str | None = None
    package: str | None = None
    errors: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.available:
            return "READY"
        if self.configured:
            return "NOT_READY"
        return "NOT_CONFIGURED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "configured": self.configured,
            "available": self.available,
            "path": self.path,
            "revision": self.revision,
            "expected_revision": self.expected_revision,
            "python": self.python,
            "package": self.package,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ExternalRunResult:
    stage: str
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    workspace: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    manifest_path: Path | None = None

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0


@dataclass(frozen=True)
class TrainingExecution:
    checkpoint_path: Path
    iteration: int
    metrics: list[dict[str, Any]] = field(default_factory=list)
    result: ExternalRunResult | None = None


@dataclass(frozen=True)
class Sim2SimExecution:
    seed: int
    status: str
    exit_code: int
    duration_seconds: float
    metrics: dict[str, float]
    command: list[str]
    artifacts: dict[str, Path] = field(default_factory=dict)
    stderr: str = ""
