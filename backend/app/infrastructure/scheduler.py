"""Small, durable resource scheduler primitives for Local File Mode."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from backend.app.infrastructure.local_file import FileLock
from backend.app.domain.contracts import RunEvent
from backend.app.domain.state_machine import RunStatus


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_c: float | None = None

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)


def collect_gpu_snapshots() -> list[GpuSnapshot]:
    """Read lightweight GPU metrics; return an empty list when unavailable."""

    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    snapshots: list[GpuSnapshot] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            continue
        try:
            temperature = None if fields[6].lower() in {"n/a", "[not supported]"} else float(fields[6])
            snapshots.append(GpuSnapshot(index=int(fields[0]), uuid=fields[1], name=fields[2], memory_total_mb=int(float(fields[3])), memory_used_mb=int(float(fields[4])), utilization_percent=float(fields[5]), temperature_c=temperature))
        except ValueError:
            continue
    return snapshots


@dataclass(frozen=True)
class GpuLease:
    lease_id: str
    gpu_uuid: str
    owner_id: str
    run_id: str
    memory_gb: float
    exclusive: bool
    created_at: float


class LocalGpuScheduler:
    """File-backed lease allocator for one local host.

    It intentionally does not start training processes.  Worker adapters own
    process execution; this class only makes admission decisions and gives
    them a recoverable lease file.
    """

    def __init__(self, root: Path, *, max_jobs_per_gpu: int = 3, utilization_limit: float = 90.0, memory_limit: float = 90.0) -> None:
        self.root = Path(root).expanduser().resolve()
        self.leases_root = self.root / "leases"
        self.leases_root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(self.root / "scheduler.lock")
        self.max_jobs_per_gpu = max_jobs_per_gpu
        self.utilization_limit = utilization_limit
        self.memory_limit = memory_limit

    def acquire(self, *, run_id: str, owner_id: str, memory_gb: float, exclusive: bool = False, snapshots: list[GpuSnapshot] | None = None) -> GpuLease | None:
        if memory_gb <= 0:
            raise ValueError("memory_gb must be positive")
        snapshots = snapshots if snapshots is not None else collect_gpu_snapshots()
        with self.lock:
            leases = self._read_leases()
            for gpu in snapshots:
                current = [lease for lease in leases if lease.gpu_uuid == gpu.uuid]
                if any(lease.exclusive for lease in current) or (exclusive and current):
                    continue
                if len(current) >= self.max_jobs_per_gpu:
                    continue
                if gpu.memory_total_mb and (gpu.memory_used_mb / gpu.memory_total_mb) * 100 >= self.memory_limit:
                    continue
                if gpu.utilization_percent >= self.utilization_limit:
                    continue
                reserved_gb = sum(lease.memory_gb for lease in current)
                if reserved_gb + memory_gb > gpu.memory_free_mb / 1024:
                    continue
                lease = GpuLease(lease_id=str(uuid.uuid4()), gpu_uuid=gpu.uuid, owner_id=owner_id, run_id=run_id, memory_gb=memory_gb, exclusive=exclusive, created_at=time.time())
                self._write_leases(leases + [lease])
                return lease
        return None

    def release(self, lease_id: str) -> bool:
        with self.lock:
            leases = self._read_leases()
            remaining = [lease for lease in leases if lease.lease_id != lease_id]
            if len(remaining) == len(leases):
                return False
            self._write_leases(remaining)
            return True

    def recover(self, *, max_age_seconds: float = 120.0) -> int:
        now = time.time()
        with self.lock:
            leases = self._read_leases()
            remaining = [lease for lease in leases if now - lease.created_at <= max_age_seconds]
            self._write_leases(remaining)
            return len(leases) - len(remaining)

    def _read_leases(self) -> list[GpuLease]:
        leases: list[GpuLease] = []
        for path in sorted(self.leases_root.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                leases.append(GpuLease(lease_id=str(value["lease_id"]), gpu_uuid=str(value["gpu_uuid"]), owner_id=str(value["owner_id"]), run_id=str(value["run_id"]), memory_gb=float(value["memory_gb"]), exclusive=bool(value["exclusive"]), created_at=float(value["created_at"])))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return leases

    def _write_leases(self, leases: list[GpuLease]) -> None:
        known = {path.stem: path for path in self.leases_root.glob("*.json")}
        wanted = {lease.lease_id for lease in leases}
        for lease_id, path in known.items():
            if lease_id not in wanted:
                path.unlink(missing_ok=True)
        for lease in leases:
            path = self.leases_root / f"{lease.lease_id}.json"
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{lease.lease_id}.", suffix=".tmp", dir=self.leases_root)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(json.dumps({"schema_version": "gpu_lease.v1", "hostname": socket.gethostname(), **lease.__dict__}, sort_keys=True, indent=2) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)


class LocalRunRecovery:
    """Reconcile runs whose local worker process disappeared."""

    active_statuses = frozenset({RunStatus.TRAINING, RunStatus.EXPORTING, RunStatus.SIM2SIM_RUNNING})

    def __init__(self, runtime_root: Path, *, stale_after_seconds: float = 120.0) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.stale_after_seconds = stale_after_seconds

    def reconcile(self, uow, *, now: float | None = None) -> list[str]:
        """Mark stale active runs ``INTERRUPTED`` and append a recovery event."""

        current_time = time.time() if now is None else now
        recovered: list[str] = []
        with uow:
            for run_id, run in list(uow.runs._runs.items()):
                if run.status not in self.active_statuses:
                    continue
                attempt = uow.runs.attempt(run.current_attempt_id)
                if attempt is None or self._attempt_alive(run_id, attempt, current_time):
                    continue
                updated_run = run.model_copy(update={"status": RunStatus.INTERRUPTED, "updated_at": _iso_now()})
                updated_attempt = attempt.model_copy(update={"status": RunStatus.INTERRUPTED, "finished_at": _iso_now()})
                uow.runs.update(updated_run)
                uow.runs.update_attempt(updated_attempt)
                prior_events = uow.events.list_after(run_id, attempt.attempt_id, 0)
                event = RunEvent(seq=len(prior_events) + 1, run_id=run_id, attempt_id=attempt.attempt_id, event_type="system", stage="recovery", level="WARNING", message="Run interrupted after local worker loss", payload={"requeued": False}, created_at=_iso_now())
                uow.events.append(event)
                recovered.append(run_id)
        return recovered

    def _attempt_alive(self, run_id: str, attempt, now: float) -> bool:
        marker = self.runtime_root / "runs" / run_id / "process.json"
        if marker.exists():
            try:
                value = json.loads(marker.read_text(encoding="utf-8"))
                pid = int(value.get("pid", 0))
                if pid > 0 and _pid_alive(pid):
                    heartbeat = float(value.get("heartbeat_at", now))
                    return now - heartbeat <= self.stale_after_seconds
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        heartbeat_value = attempt.last_heartbeat_at or ""
        try:
            from datetime import datetime

            heartbeat = datetime.fromisoformat(heartbeat_value).timestamp()
        except (TypeError, ValueError):
            return False
        return now - heartbeat <= self.stale_after_seconds


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError, SystemError):
        return False
    return True


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = ["GpuLease", "GpuSnapshot", "LocalGpuScheduler", "LocalRunRecovery", "collect_gpu_snapshots"]
