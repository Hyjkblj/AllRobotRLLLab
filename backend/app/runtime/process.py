"""Safe subprocess execution and output manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from backend.app.runtime.contracts import ExternalRunResult, RunnerError


def command_from_env(name: str, *, default: Sequence[str] | None = None) -> tuple[str, ...] | None:
    raw = os.getenv(name, "").strip()
    if raw:
        try:
            parsed = shlex.split(raw, posix=False if os.name == "nt" else True)
            values = tuple(item[1:-1] if len(item) >= 2 and item[0] == item[-1] and item[0] in {"'", '"'} else item for item in parsed)
        except ValueError as exc:
            raise RunnerError("RUNTIME_COMMAND_INVALID", f"{name} is not a valid argument list: {exc}") from exc
        if values:
            return values
    return tuple(str(item) for item in default) if default else None


def run_external(*, stage: str, workspace: Path, command: Sequence[str], timeout_seconds: float = 3600, env: dict[str, str] | None = None) -> ExternalRunResult:
    normalized = tuple(str(item) for item in command)
    if not normalized or any(not item for item in normalized):
        raise RunnerError("RUNTIME_COMMAND_INVALID", f"{stage} command must be a non-empty argument list")
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    for directory in ("input", "outputs", "logs", "metrics", "reports", "manifest"):
        (workspace / directory).mkdir(exist_ok=True)
    try:
        process = subprocess.run(
            normalized,
            cwd=workspace,
            env={**os.environ, **(env or {}), "PLATFORM_STAGE": stage},
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RunnerError("RUNTIME_EXECUTABLE_NOT_FOUND", f"{stage} executable was not found: {normalized[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RunnerError("RUNTIME_TIMEOUT", f"{stage} exceeded {timeout_seconds:g}s timeout") from exc
    result = ExternalRunResult(stage=stage, command=normalized, return_code=process.returncode, stdout=process.stdout or "", stderr=process.stderr or "", workspace=workspace)
    (workspace / "logs" / f"{stage}.json").write_text(json.dumps({"stage": stage, "command": list(normalized), "return_code": result.return_code, "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result.succeeded:
        raise RunnerError("RUNTIME_PROCESS_FAILED", f"{stage} process exited with code {result.return_code}", details={"return_code": result.return_code, "stderr": result.stderr[-4000:], "command": list(normalized)})
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_output_manifest(root: Path, *, stage: str, outputs: Iterable[Path], metadata: dict | None = None) -> Path:
    root = root.resolve()
    records = []
    for path in sorted((item.resolve() for item in outputs), key=lambda item: str(item)):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RunnerError("RUNTIME_OUTPUT_OUTSIDE_WORKSPACE", f"output is outside workspace: {path}") from exc
        records.append({"path": relative, "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
    if not records:
        raise RunnerError("RUNTIME_OUTPUT_MISSING", f"{stage} produced no output files")
    manifest = root / "manifest" / f"{stage}.json"
    manifest.write_text(json.dumps({"schema_version": "external_stage_manifest.v1", "stage": stage, "outputs": records, "metadata": metadata or {}}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["command_from_env", "file_sha256", "run_external", "write_output_manifest"]
