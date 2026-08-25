"""Idempotent, subprocess-safe stage runner foundation.

Celery/RQ adapters can call this runner later.  The runner itself has no
queue/database dependency and always records the command as an argument list.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def idempotency_key(*, input_hash: str, processor_version: str, config_hash: str, robot_version: str) -> str:
    payload = "|".join((input_hash, processor_version, config_hash, robot_version)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StageExecutionContext:
    stage: str
    workspace: Path
    input_hash: str
    processor_version: str
    config_hash: str
    robot_version: str

    @property
    def key(self) -> str:
        return idempotency_key(input_hash=self.input_hash, processor_version=self.processor_version, config_hash=self.config_hash, robot_version=self.robot_version)

    def prepare(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        for directory in ("input", "work", "outputs", "logs", "metrics", "reports", "manifest"):
            (self.workspace / directory).mkdir(exist_ok=True)


@dataclass(frozen=True)
class StageResult:
    stage: str
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    idempotency_key: str

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0

    def write_log(self, path: Path) -> None:
        path.write_text(json.dumps({"stage": self.stage, "command": list(self.command), "return_code": self.return_code, "stdout": self.stdout, "stderr": self.stderr, "idempotency_key": self.idempotency_key}, ensure_ascii=False, indent=2), encoding="utf-8")


class StageRunner:
    """Execute a fixed command in a generated, bounded workspace."""

    def run(self, context: StageExecutionContext, command: Sequence[str], *, timeout_seconds: float = 3600) -> StageResult:
        context.prepare()
        normalized = tuple(str(item) for item in command)
        if not normalized or any(not item for item in normalized):
            raise ValueError("stage command must be a non-empty argument list")
        workspace = context.workspace.resolve()
        process = subprocess.run(
            normalized,
            cwd=workspace,
            env={**os.environ, "PLATFORM_STAGE": context.stage, "PLATFORM_IDEMPOTENCY_KEY": context.key},
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return StageResult(context.stage, normalized, process.returncode, process.stdout, process.stderr, context.key)


__all__ = ["StageExecutionContext", "StageResult", "StageRunner", "idempotency_key"]

