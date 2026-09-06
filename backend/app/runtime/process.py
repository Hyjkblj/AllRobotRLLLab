"""Safe subprocess execution and output manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from backend.app.runtime.contracts import ExternalRunResult, RunnerError


_process_context: contextvars.ContextVar[tuple[str, Path] | None] = contextvars.ContextVar(
    "allrobotrl_external_process_context", default=None
)


@contextmanager
def external_run_context(*, run_id: str, runtime_root: Path) -> Iterator[None]:
    """Associate external subprocesses with a durable Run marker.

    The context is deliberately process-local so runner APIs remain backwards
    compatible with existing adapters and tests. ``run_external`` writes the
    marker only while a real child process is alive; API cancellation can then
    terminate that child without ever targeting the worker process itself.
    """

    token = _process_context.set((str(run_id), Path(runtime_root).expanduser().resolve()))
    try:
        yield
    finally:
        _process_context.reset(token)


def _marker_path(run_id: str, runtime_root: Path) -> Path:
    value = str(run_id)
    # Run IDs are generated UUID-like opaque identifiers. Reject path
    # separators here so an API path or malformed manifest can never redirect
    # process control outside the runtime root.
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError("run_id is not a safe process-marker identifier")
    return runtime_root / "runs" / value / "external-process.json"


def _write_process_marker(path: Path, *, pid: int, stage: str, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "external_process_marker.v1",
        "pid": int(pid),
        "stage": stage,
        "command": [str(item) for item in command],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _remove_process_marker(path: Path, *, pid: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("pid", -1)) != int(pid):
            return
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # A cancellation request may have removed or replaced the marker.
        pass
    path.unlink(missing_ok=True)


def terminate_run_process(run_id: str, runtime_root: Path) -> bool:
    """Terminate the external child process registered for ``run_id``.

    Returns ``True`` only when a live child was found and a termination signal
    was sent. Stale or malformed markers are removed and treated as no-op.
    """

    try:
        marker = _marker_path(str(run_id), Path(runtime_root).expanduser().resolve())
    except ValueError:
        return False
    try:
        payload: dict[str, Any] = json.loads(marker.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        marker.unlink(missing_ok=True)
        return False
    if pid <= 0:
        marker.unlink(missing_ok=True)
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError, SystemError):
        marker.unlink(missing_ok=True)
        return False
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, timeout=10, check=False)
        else:
            # ``start_new_session=True`` makes the child its own process group,
            # so descendants (Isaac/MuJoCo workers) are terminated together.
            os.killpg(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, PermissionError, SystemError):
            marker.unlink(missing_ok=True)
            return False
    return True


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
    context = _process_context.get()
    marker = _marker_path(*context) if context is not None else None
    process = None
    stdout = ""
    stderr = ""
    try:
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        process = subprocess.Popen(
            normalized,
            cwd=workspace,
            env={**os.environ, **(env or {}), "PLATFORM_STAGE": stage},
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name != "nt",
            creationflags=creation_flags,
        )
        if marker is not None:
            _write_process_marker(marker, pid=process.pid, stage=stage, command=normalized)
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise RunnerError("RUNTIME_EXECUTABLE_NOT_FOUND", f"{stage} executable was not found: {normalized[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, text=True, timeout=10, check=False)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except (OSError, subprocess.SubprocessError):
                process.kill()
            stdout, stderr = process.communicate()
        raise RunnerError("RUNTIME_TIMEOUT", f"{stage} exceeded {timeout_seconds:g}s timeout") from exc
    finally:
        if marker is not None and process is not None:
            _remove_process_marker(marker, pid=process.pid)
    if process is None:
        raise RunnerError("RUNTIME_PROCESS_FAILED", f"{stage} process was not started")
    result = ExternalRunResult(stage=stage, command=normalized, return_code=process.returncode, stdout=stdout or "", stderr=stderr or "", workspace=workspace)
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


__all__ = ["command_from_env", "external_run_context", "file_sha256", "run_external", "terminate_run_process", "write_output_manifest"]
