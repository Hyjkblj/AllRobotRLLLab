"""Executable motion asset pipeline for Local File Mode.

The production GPU processors (GVHMR/GMR) are intentionally not faked here.
Direct G1 joint trajectories are converted on CPU so the rest of the platform
can be exercised end to end; video/human-pose inputs fail with an explicit
runtime-unavailable code until those processors are installed on a worker.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.adapters.motion import MotionDetectionError, MotionSourceRegistry
from backend.app.adapters.motion.detector import sha256_file
from backend.app.application.asset_service import AssetService
from backend.app.application.motion_editor import MotionEditor
from backend.app.application.run_service import RunServiceError, utc_now
from backend.app.domain.contracts import (
    Actor,
    AssetKind,
    AssetRecord,
    AssetVersion,
    AssetVersionStatus,
    ArrayField,
    LicenseInfo,
    MotionEditConfig,
    MotionPipelineRecord,
    MotionPipelineStage,
    MotionQuality,
    MotionQualityReport,
    ProjectRole,
    RetargetMotion,
    SchemaVersion,
    SourceMotionDescriptor,
    TaskSubmission,
    TrainMotionNPZ,
    ValidationIssue,
    ValidationSeverity,
)
from backend.app.domain.motion import MotionArrays
from backend.app.infrastructure.local_file import FileLock
from backend.app.infrastructure.memory import RepositoryConflict
from backend.app.infrastructure.object_store import LocalObjectStore
from backend.app.runtime.contracts import RunnerError


class MotionPipelineError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class MotionPipelineStore:
    """Small durable JSON store shared by API and the local worker process."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self._lock = threading.RLock()
        self._records: dict[str, MotionPipelineRecord] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            self._file_lock = FileLock(self.root / "store.lock")
        else:
            self._file_lock = None

    def get(self, pipeline_id: str) -> MotionPipelineRecord | None:
        with self._guard():
            if self.root is None:
                return self._records.get(pipeline_id)
            path = self.root / f"{pipeline_id}.json"
            if not path.exists():
                return None
            try:
                return MotionPipelineRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise MotionPipelineError("PIPELINE_STATE_INVALID", f"invalid pipeline state: {path}", status_code=500) from exc

    def list_for_source(self, asset_version_id: str) -> list[MotionPipelineRecord]:
        with self._guard():
            if self.root is None:
                records = list(self._records.values())
            else:
                records = []
                for path in sorted(self.root.glob("*.json")):
                    try:
                        records.append(MotionPipelineRecord.model_validate_json(path.read_text(encoding="utf-8")))
                    except Exception:
                        continue
            return sorted((record for record in records if record.source_asset_version_id == asset_version_id), key=lambda item: item.created_at, reverse=True)

    def save(self, record: MotionPipelineRecord) -> MotionPipelineRecord:
        with self._guard():
            if self.root is None:
                self._records[record.pipeline_id] = record
                return record
            path = self.root / f"{record.pipeline_id}.json"
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{record.pipeline_id}.", suffix=".tmp", dir=self.root)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(record.model_dump_json(indent=2))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            return record

    class _Guard:
        def __init__(self, owner: "MotionPipelineStore") -> None:
            self.owner = owner

        def __enter__(self):
            self.owner._lock.acquire()
            if self.owner._file_lock is not None:
                self.owner._file_lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.owner._file_lock is not None:
                self.owner._file_lock.release()
            self.owner._lock.release()

    def _guard(self):
        return self._Guard(self)


class MotionPipelineService:
    stage_names = ("detect", "retarget", "edit", "compile", "publish")

    def __init__(self, *, uow, object_store, robot_adapter, motion_registry: MotionSourceRegistry, motion_editor: MotionEditor, asset_service: AssetService, store: MotionPipelineStore, task_dispatcher=None, kinematics_compiler=None, gvhmr_runner=None, gmr_runner=None) -> None:
        self.uow = uow
        self.object_store = object_store
        self.robot_adapter = robot_adapter
        self.motion_registry = motion_registry
        self.motion_editor = motion_editor
        self.asset_service = asset_service
        self.store = store
        self.task_dispatcher = task_dispatcher
        self.kinematics_compiler = kinematics_compiler
        self.gvhmr_runner = gvhmr_runner
        self.gmr_runner = gmr_runner

    def submit(self, *, actor: Actor, asset_version_id: str, edit_config: MotionEditConfig | None = None, sync: bool = False) -> tuple[MotionPipelineRecord, TaskSubmission | None]:
        version, asset = self._asset_for_actor(asset_version_id, actor)
        if asset.kind not in {AssetKind.MOTION, AssetKind.VIDEO}:
            raise MotionPipelineError("ASSET_KIND_INVALID", "motion processing requires a motion or video asset", status_code=409)
        if version.status != AssetVersionStatus.READY:
            raise MotionPipelineError("ASSET_NOT_READY", f"asset version must be READY before processing, got {version.status}", status_code=409)
        config = edit_config or MotionEditConfig(source_motion_version_id=asset_version_id, robot_id=self.robot_adapter.name)
        if config.source_motion_version_id != asset_version_id:
            raise MotionPipelineError("MOTION_SOURCE_MISMATCH", "edit config source_motion_version_id must match the asset version")
        if config.robot_id != self.robot_adapter.name:
            raise MotionPipelineError("ROBOT_NOT_FOUND", f"unknown robot adapter: {config.robot_id}", status_code=404)
        existing = next((item for item in self.store.list_for_source(asset_version_id) if item.edit_config and item.edit_config.model_dump(mode="json") == config.model_dump(mode="json") and item.status in {"QUEUED", "RUNNING", "READY"}), None)
        if existing is not None:
            return existing, None
        now = utc_now()
        record = MotionPipelineRecord(
            pipeline_id=str(uuid.uuid4()),
            project_id=asset.project_id,
            source_asset_version_id=asset_version_id,
            stages=[MotionPipelineStage(name=name) for name in self.stage_names],
            edit_config=config,
            created_at=now,
            updated_at=now,
        )
        self.store.save(record)
        if sync:
            return self.process(record.pipeline_id), None
        if self.task_dispatcher is None:
            raise MotionPipelineError("TASK_QUEUE_UNAVAILABLE", "motion worker queue is not configured", status_code=503)
        payload = {"pipeline_id": record.pipeline_id, "asset_version_id": asset_version_id}
        queue = "motion-gpu" if asset.kind == AssetKind.VIDEO else "motion-cpu"
        key = hashlib.sha256(json.dumps(payload | {"edit_config": config.model_dump(mode="json")}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        try:
            task_id = self.task_dispatcher.enqueue(queue=queue, task="allrobotrl.motion.process", payload=payload, idempotency_key=key)
        except Exception as exc:
            self._fail(record, "TASK_QUEUE_UNAVAILABLE", str(exc))
            raise MotionPipelineError("TASK_QUEUE_UNAVAILABLE", str(exc), status_code=503) from exc
        record = record.model_copy(update={"task_id": task_id, "updated_at": utc_now()})
        self.store.save(record)
        return record, TaskSubmission(task_id=task_id, run_id=record.pipeline_id, attempt_id=record.pipeline_id, operation="motion_process", queue=queue, idempotency_key=key)

    def get(self, *, actor: Actor, pipeline_id: str) -> MotionPipelineRecord:
        record = self.store.get(pipeline_id)
        if record is None:
            raise MotionPipelineError("MOTION_PIPELINE_NOT_FOUND", f"motion pipeline not found: {pipeline_id}", status_code=404)
        with self.uow:
            member = self.uow.projects.member(record.project_id, actor.user_id)
        if member is None:
            raise MotionPipelineError("PROJECT_ACCESS_DENIED", "user is not a member of this project", status_code=403)
        return record

    def get_for_asset(self, *, actor: Actor, asset_version_id: str) -> MotionPipelineRecord | None:
        _version, _asset = self._asset_for_actor(asset_version_id, actor, minimum_viewer=True)
        records = self.store.list_for_source(asset_version_id)
        return records[0] if records else None

    def validate_asset_version(self, asset_version_id: str) -> AssetVersion:
        """Validate an uploaded object in a worker-safe, idempotent operation."""
        with self.uow:
            version = self.uow.assets.version(asset_version_id)
            asset = self.uow.assets.get(version.asset_id) if version else None
        if version is None or asset is None:
            raise MotionPipelineError("ASSET_VERSION_NOT_FOUND", f"asset version not found: {asset_version_id}", status_code=404)
        if version.status not in {AssetVersionStatus.VALIDATING, AssetVersionStatus.UPLOADED}:
            return version
        try:
            path = self._resolve_source(version)
            metadata = self.object_store.stat(version.object_key)
            if asset.kind == AssetKind.MOTION:
                self.motion_registry.detect(path, asset_version_id=asset_version_id, trusted_pickle=False)
            elif asset.kind == AssetKind.VIDEO:
                if self.gvhmr_runner is None:
                    raise MotionDetectionError("GVHMR_RUNTIME_UNAVAILABLE", "video processing requires the GVHMR Linux GPU runtime")
                if path.stat().st_size == 0:
                    raise MotionDetectionError("SCHEMA_INVALID", "video object is empty")
            elif asset.kind == AssetKind.MODEL:
                if path.stat().st_size == 0:
                    raise MotionDetectionError("SCHEMA_INVALID", "model object is empty")
            digest = str(metadata.get("sha256") or sha256_file(path))
            return self.asset_service.mark_validated(asset_version_id=asset_version_id, valid=True, sha256=digest, size_bytes=int(metadata.get("size_bytes") or path.stat().st_size))
        except MotionDetectionError as exc:
            return self.asset_service.mark_validated(asset_version_id=asset_version_id, valid=False, rejection_code=exc.code)
        except Exception as exc:
            return self.asset_service.mark_validated(asset_version_id=asset_version_id, valid=False, rejection_code="ASSET_OBJECT_UNAVAILABLE")

    def process(self, pipeline_id: str) -> MotionPipelineRecord:
        record = self.store.get(pipeline_id)
        if record is None:
            raise MotionPipelineError("MOTION_PIPELINE_NOT_FOUND", f"motion pipeline not found: {pipeline_id}", status_code=404)
        if record.status == "READY":
            return record
        record = record.model_copy(update={"status": "RUNNING", "updated_at": utc_now()})
        self.store.save(record)
        try:
            version, asset = self._asset_without_actor(record.source_asset_version_id)
            if version.status != AssetVersionStatus.READY:
                raise MotionPipelineError("ASSET_NOT_READY", f"asset version must be READY, got {version.status}", status_code=409)
            path = self._resolve_source(version)
            record = self._stage(record, "detect", "RUNNING")
            generated_retarget = None
            try:
                descriptor = self.motion_registry.detect(path, asset_version_id=version.asset_version_id, trusted_pickle=False)
            except MotionDetectionError as exc:
                if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"} or self.gvhmr_runner is None or self.gmr_runner is None:
                    raise
                descriptor = SourceMotionDescriptor(asset_version_id=version.asset_version_id, file_format="pt", detected_type="gvhmr_result", source_skeleton="smpl", coord_frame="world_z_up", quaternion_convention="xyzw", license=LicenseInfo(status="declared", source="user", processing_scope="platform motion processing"), detector_version="gvhmr-gmr-pipeline.v1")
                record = self._stage(record, "detect", "SUCCEEDED", message="video accepted by GVHMR/GMR runtime", source_descriptor=descriptor)
                record = self._stage(record, "retarget", "RUNNING")
                try:
                    gvhmr_output, _ = self.gvhmr_runner.run(video_path=path, output_dir=(self.store.root or Path(tempfile.gettempdir())) / "external" / record.pipeline_id / "gvhmr")
                    generated_retarget, _ = self.gmr_runner.run(source_path=gvhmr_output, output_dir=(self.store.root or Path(tempfile.gettempdir())) / "external" / record.pipeline_id / "gmr", robot="unitree_g1")
                except RunnerError as runner_exc:
                    raise MotionPipelineError(runner_exc.code, str(runner_exc), status_code=503) from runner_exc
                record = self._stage(record, "retarget", "SUCCEEDED", message="GVHMR/GMR retarget complete")
            record = self._stage(record, "detect", "SUCCEEDED", message=descriptor.detected_type, source_descriptor=descriptor)
            if generated_retarget is None:
                record = self._stage(record, "retarget", "RUNNING")
            if descriptor.detected_type not in {"g1_joint_trajectory", "gvhmr_result"}:
                raise MotionPipelineError("RETARGET_RUNTIME_UNAVAILABLE", "human-pose/GVHMR retargeting requires the Linux GPU processor", status_code=503)
            expected_names = list(self.robot_adapter.get_spec().joint_names)
            if descriptor.joint_names and descriptor.joint_names not in (expected_names, expected_names + ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]):
                raise MotionPipelineError("MOTION_JOINT_ORDER_MISMATCH", "source joint names do not match the locked G1 adapter order", status_code=422)
            arrays = self._load_arrays(generated_retarget or path, descriptor)
            retarget = self._retarget_metadata(arrays, descriptor, version)
            record = self._stage(record, "retarget", "SUCCEEDED", message="direct G1 trajectory", retarget_motion=retarget)
            record = self._stage(record, "edit", "RUNNING")
            edit = record.edit_config or MotionEditConfig(source_motion_version_id=version.asset_version_id, robot_id=self.robot_adapter.name)
            edited = self.motion_editor.apply(arrays, edit)
            if edited.arrays is None or not edited.validation.valid:
                raise MotionPipelineError("MOTION_QUALITY_BLOCKED", "motion edit quality gate blocked compilation", status_code=422)
            quality = edited.quality
            if self.kinematics_compiler is None:
                quality = edited.quality.model_copy(update={"status": "WARNING", "issues": [*edited.quality.issues, ValidationIssue(code="KINEMATICS_APPROXIMATION", message="body arrays use root pose until the MuJoCo kinematics compiler is installed", severity=ValidationSeverity.WARNING, suggested_action="configure G1_MJCF_PATH and the MuJoCo worker before real-robot deployment")]})
            record = self._stage(record, "edit", "SUCCEEDED", message=quality.status, quality=quality)
            record = self._stage(record, "compile", "RUNNING")
            train_motion, output_path = self._compile(edited.arrays, version, asset, edited.quality)
            record = self._stage(record, "compile", "SUCCEEDED", message="TrainMotionNPZ generated", train_motion=train_motion)
            record = self._stage(record, "publish", "RUNNING")
            output_version, output_key = self._publish(output_path, train_motion, asset)
            record = self._stage(record, "publish", "SUCCEEDED", message="output asset is READY", output_asset_version_id=output_version.asset_version_id, output_object_key=output_key)
            ready = record.model_copy(update={"status": "READY", "updated_at": utc_now()})
            self.store.save(ready)
            return ready
        except (MotionPipelineError, MotionDetectionError) as exc:
            code = exc.code if isinstance(exc, (MotionPipelineError, MotionDetectionError)) else "MOTION_PIPELINE_FAILED"
            return self._fail(record, code, str(exc))
        except Exception as exc:
            return self._fail(record, "MOTION_PIPELINE_FAILED", str(exc))

    def _asset_for_actor(self, asset_version_id: str, actor: Actor, minimum_viewer: bool = False) -> tuple[AssetVersion, AssetRecord]:
        with self.uow:
            version = self.uow.assets.version(asset_version_id)
            asset = self.uow.assets.get(version.asset_id) if version else None
            if version is None or asset is None:
                raise MotionPipelineError("ASSET_VERSION_NOT_FOUND", f"asset version not found: {asset_version_id}", status_code=404)
            try:
                self.asset_service._require_member(asset.project_id, actor.user_id, minimum=ProjectRole.VIEWER if minimum_viewer else ProjectRole.EDITOR)
            except RunServiceError as exc:
                raise MotionPipelineError(exc.code, exc.message, status_code=exc.status_code) from exc
            return version, asset

    def _asset_without_actor(self, asset_version_id: str) -> tuple[AssetVersion, AssetRecord]:
        with self.uow:
            version = self.uow.assets.version(asset_version_id)
            asset = self.uow.assets.get(version.asset_id) if version else None
        if version is None or asset is None:
            raise MotionPipelineError("ASSET_VERSION_NOT_FOUND", f"asset version not found: {asset_version_id}", status_code=404)
        return version, asset

    def _resolve_source(self, version: AssetVersion) -> Path:
        try:
            resolver = getattr(self.object_store, "resolve_path", None)
            if callable(resolver):
                return resolver(version.object_key)
            downloader = getattr(self.object_store, "download_file", None)
            if not callable(downloader):
                raise MotionPipelineError("MOTION_SOURCE_NOT_LOCAL", "object store cannot materialize a worker input", status_code=503)
            suffix = Path(version.original_filename or version.object_key).suffix or ".motion"
            target = Path(tempfile.mkdtemp(prefix="allrobotrl-motion-")) / f"source{suffix}"
            return downloader(version.object_key, target)
        except Exception as exc:
            if isinstance(exc, MotionPipelineError):
                raise
            raise MotionPipelineError("ASSET_OBJECT_NOT_FOUND", "uploaded motion object is not available", status_code=409) from exc

    def _stage(self, record: MotionPipelineRecord, name: str, status: str, *, message: str | None = None, **updates: Any) -> MotionPipelineRecord:
        now = utc_now()
        stages = list(record.stages)
        index = next(i for i, item in enumerate(stages) if item.name == name)
        current = stages[index]
        started = current.started_at or (now if status == "RUNNING" else current.started_at)
        finished = now if status in {"SUCCEEDED", "FAILED", "SKIPPED"} else current.finished_at
        duration = None
        if started and finished:
            try:
                duration = max(0.0, (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds())
            except ValueError:
                duration = None
        stages[index] = current.model_copy(update={"status": status, "started_at": started, "finished_at": finished, "duration_seconds": duration, "message": message or current.message, "error_code": updates.pop("error_code", current.error_code)})
        record = record.model_copy(update={"stages": stages, "updated_at": now, **updates})
        self.store.save(record)
        return record

    def _fail(self, record: MotionPipelineRecord, code: str, message: str) -> MotionPipelineRecord:
        stages = list(record.stages)
        running = next((i for i, item in enumerate(stages) if item.status == "RUNNING"), None)
        if running is not None:
            current = stages[running]
            stages[running] = current.model_copy(update={"status": "FAILED", "finished_at": utc_now(), "message": message, "error_code": code})
        failed = record.model_copy(update={"status": "FAILED", "stages": stages, "error_code": code, "error_message": message, "updated_at": utc_now()})
        self.store.save(failed)
        return failed

    def _load_arrays(self, path: Path, descriptor: SourceMotionDescriptor) -> MotionArrays:
        suffix = path.suffix.lower()
        values: np.ndarray
        fps = 30.0
        root_pos = None
        root_rot = None
        if suffix == ".npz":
            archive = np.load(path, allow_pickle=False)
            try:
                key = next(name for name in ("joint_pos", "dof_pos", "qpos") if name in archive.files)
                values = np.asarray(archive[key], dtype=np.float64)
                if "fps" in archive.files:
                    fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
                root_pos = np.asarray(archive["root_pos"], dtype=np.float64) if "root_pos" in archive.files else None
                root_rot = np.asarray(archive["root_rot"], dtype=np.float64) if "root_rot" in archive.files else None
            finally:
                archive.close()
        elif suffix == ".csv":
            rows: list[list[float]] = []
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                first = next(reader, [])
                header = next(reader, []) if first and first[0].lstrip().lower().startswith("# joint_names") else first
                values_start = 2 if header[:2] == ["time_s", "phase"] else 0
                for row in reader:
                    values = row[values_start:] if values_start else row[-29:]
                    rows.append([float(value) for value in values[-29:]])
            values = np.asarray(rows, dtype=np.float64)
        elif suffix == ".pt":
            try:
                import torch
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except ImportError as exc:
                raise MotionPipelineError("PROCESSOR_UNAVAILABLE", "torch is required to process .pt files", status_code=503) from exc
            except Exception as exc:
                raise MotionPipelineError("SCHEMA_INVALID", f"safe .pt loading failed: {exc}") from exc
            if hasattr(payload, "detach"):
                payload = {"joint_pos": payload}
            if not isinstance(payload, dict):
                raise MotionPipelineError("SCHEMA_INVALID", ".pt must contain a tensor or tensor dictionary")
            key = next((name for name in ("joint_pos", "dof_pos", "qpos") if name in payload), None)
            if key is None:
                raise MotionPipelineError("SCHEMA_INVALID", "joint trajectory field is missing")
            values = payload[key].detach().cpu().numpy().astype(np.float64)
        elif suffix == ".pkl":
            import pickle
            try:
                with path.open("rb") as stream:
                    payload = pickle.load(stream)
            except Exception as exc:
                raise MotionPipelineError("SCHEMA_INVALID", f"trusted GMR output could not be loaded: {exc}") from exc
            if not isinstance(payload, dict):
                raise MotionPipelineError("SCHEMA_INVALID", "GMR output must be a dictionary")
            key = next((name for name in ("dof_pos", "joint_pos", "qpos") if name in payload), None)
            if key is None:
                raise MotionPipelineError("SCHEMA_INVALID", "GMR output is missing dof_pos")
            values = np.asarray(payload[key], dtype=np.float64)
            fps = float(payload.get("fps", 30.0))
            root_pos = np.asarray(payload.get("root_pos"), dtype=np.float64) if payload.get("root_pos") is not None else None
            root_rot = np.asarray(payload.get("root_rot"), dtype=np.float64) if payload.get("root_rot") is not None else None
        else:
            raise MotionPipelineError("UNSUPPORTED_SOURCE_TYPE", f"unsupported motion extension: {suffix}")
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] not in (29, 36):
            raise MotionPipelineError("SCHEMA_INVALID", "G1 trajectory must have shape [N, 29] or [N, 36]")
        if values.shape[1] == 36:
            values = values[:, :29]
        names = tuple(self.robot_adapter.get_spec().joint_names)
        count = len(values)
        if root_pos is None or np.asarray(root_pos).shape != (count, 3):
            root_pos = np.zeros((count, 3), dtype=np.float64)
            root_pos[:, 2] = 0.75
        if root_rot is None or np.asarray(root_rot).shape != (count, 4):
            root_rot = np.zeros((count, 4), dtype=np.float64)
            root_rot[:, 3] = 1.0
        return MotionArrays(fps=fps, joint_pos=values, root_pos=np.asarray(root_pos, dtype=np.float64), root_rot=np.asarray(root_rot, dtype=np.float64), joint_names=names, quat_convention="xyzw", coord_frame=descriptor.coord_frame or "world_z_up")

    def _retarget_metadata(self, arrays: MotionArrays, descriptor: SourceMotionDescriptor, version: AssetVersion) -> RetargetMotion:
        root_rot = np.asarray(arrays.root_rot, dtype=np.float64)
        norms = np.linalg.norm(root_rot, axis=1) if len(root_rot) else np.asarray([1.0])
        quality = MotionQuality(nan_count=int(sum(np.count_nonzero(~np.isfinite(np.asarray(item))) for item in (arrays.joint_pos, arrays.root_pos, arrays.root_rot))), quat_norm_max_error=float(np.max(np.abs(norms - 1.0))), joint_limit_violation_ratio=0.0, foot_sliding_ratio=0.0)
        return RetargetMotion(robot_id=self.robot_adapter.name, fps=arrays.fps, frame_count=len(arrays.joint_pos), arrays={"dof_pos": "retarget/dof_pos.npy", "root_pos": "retarget/root_pos.npy", "root_rot": "retarget/root_rot.npy"}, array_meta={"dof_pos": {"dtype": "float32", "shape": list(arrays.joint_pos.shape)}, "root_pos": {"dtype": "float32", "shape": list(arrays.root_pos.shape)}, "root_rot": {"dtype": "float32", "shape": list(arrays.root_rot.shape), "convention": "xyzw"}}, joint_names=list(arrays.joint_names), coord_frame=arrays.coord_frame, source={"asset_version_id": version.asset_version_id, "sha256": version.sha256 or sha256_file(self._resolve_source(version))}, quality=quality, converter={"name": "g1-direct-trajectory", "version": "g1-direct-trajectory.v1"})

    def _compile(self, arrays: MotionArrays, version: AssetVersion, asset: AssetRecord, quality: MotionQualityReport) -> tuple[TrainMotionNPZ, Path]:
        workdir = (self.store.root or Path(tempfile.gettempdir())) / "outputs"
        workdir.mkdir(parents=True, exist_ok=True)
        output_path = workdir / f"{uuid.uuid4()}.npz"
        joint_pos = np.asarray(arrays.joint_pos, dtype=np.float32)
        body_names = list(self.robot_adapter.get_spec().body_names)
        if self.kinematics_compiler is not None:
            try:
                compiled = self.kinematics_compiler.compile(joint_pos=joint_pos, root_pos=np.asarray(arrays.root_pos, dtype=np.float64), root_rot=np.asarray(arrays.root_rot, dtype=np.float64), fps=arrays.fps)
            except RunnerError as exc:
                raise MotionPipelineError(exc.code, str(exc), status_code=503) from exc
        else:
            root_pos = np.asarray(arrays.root_pos, dtype=np.float32)
            root_vel = np.gradient(root_pos, 1.0 / arrays.fps, axis=0).astype(np.float32) if len(root_pos) > 1 else np.zeros_like(root_pos)
            compiled = {"joint_pos": np.asarray(joint_pos, dtype=np.float32), "joint_vel": np.gradient(joint_pos, 1.0 / arrays.fps, axis=0).astype(np.float32) if len(joint_pos) > 1 else np.zeros_like(joint_pos), "body_pos_w": np.repeat(root_pos[:, None, :], len(body_names), axis=1), "body_quat_w": np.repeat(np.asarray(arrays.root_rot, dtype=np.float32)[:, None, [3, 0, 1, 2]], len(body_names), axis=1), "body_lin_vel_w": np.repeat(root_vel[:, None, :], len(body_names), axis=1), "body_ang_vel_w": np.zeros((len(root_pos), len(body_names), 3), dtype=np.float32), "compiler_version": "g1-direct-trajectory.v1"}
        joint_pos = np.asarray(compiled["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(compiled["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(compiled["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(compiled["body_quat_w"], dtype=np.float32)
        body_lin_vel = np.asarray(compiled["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel = np.asarray(compiled["body_ang_vel_w"], dtype=np.float32)
        np.savez_compressed(output_path, joint_pos=joint_pos, joint_vel=joint_vel, body_pos_w=body_pos, body_quat_w=body_quat, body_lin_vel_w=body_lin_vel, body_ang_vel_w=body_ang_vel, fps=np.asarray(arrays.fps, dtype=np.float32), joint_names=np.asarray(arrays.joint_names), body_names=np.asarray(body_names), coord_frame=np.asarray(arrays.coord_frame), quat_convention=np.asarray("wxyz"))
        source_hash = version.sha256 or sha256_file(self._resolve_source(version))
        arrays_meta = {
            "joint_pos": ArrayField(path="joint_pos", shape=list(joint_pos.shape), dtype="float32"),
            "joint_vel": ArrayField(path="joint_vel", shape=list(joint_vel.shape), dtype="float32"),
            "body_pos_w": ArrayField(path="body_pos_w", shape=list(body_pos.shape), dtype="float32"),
            "body_quat_w": ArrayField(path="body_quat_w", shape=list(body_quat.shape), dtype="float32", convention="wxyz"),
            "body_lin_vel_w": ArrayField(path="body_lin_vel_w", shape=list(body_lin_vel.shape), dtype="float32"),
            "body_ang_vel_w": ArrayField(path="body_ang_vel_w", shape=list(body_ang_vel.shape), dtype="float32"),
        }
        train = TrainMotionNPZ(robot_id=self.robot_adapter.name, fps=arrays.fps, frame_count=len(joint_pos), joint_names=list(arrays.joint_names), body_names=body_names, arrays=arrays_meta, coord_frame=arrays.coord_frame, quat_convention="wxyz", source_motion_hash=source_hash, compiler_version=str(compiled.get("compiler_version", "g1-direct-trajectory.v1")))
        return train, output_path

    def _publish(self, output_path: Path, train_motion: TrainMotionNPZ, source_asset: AssetRecord) -> tuple[AssetVersion, str]:
        if self.object_store is None:
            raise MotionPipelineError("OBJECT_STORE_UNAVAILABLE", "object store is required to publish TrainMotionNPZ", status_code=503)
        asset_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        key = f"projects/{source_asset.project_id}/assets/{asset_id}/versions/{version_id}/train_motion.npz"
        try:
            stored = self.object_store.put_file(key, output_path, content_type="application/x-npz")
            now = utc_now()
            asset = AssetRecord(asset_id=asset_id, project_id=source_asset.project_id, kind=AssetKind.MOTION, display_name=f"{source_asset.display_name} · TrainMotionNPZ", license=source_asset.license, created_by=source_asset.created_by, created_at=now)
            version = AssetVersion(asset_version_id=version_id, asset_id=asset_id, version=1, status=AssetVersionStatus.READY, object_key=str(stored["key"]), original_filename=f"{source_asset.display_name}.train_motion.npz", content_type="application/x-npz", size_bytes=int(stored["size_bytes"]), sha256=str(stored["sha256"]), created_at=now, validated_at=now)
            with self.uow:
                self.uow.assets.create(asset, version)
            return version, str(stored["key"])
        except RepositoryConflict:
            existing = self._find_output_by_key(source_asset.project_id, key)
            if existing is not None:
                return existing, key
            raise MotionPipelineError("OUTPUT_ASSET_CONFLICT", "output asset already exists", status_code=409)

    def _find_output_by_key(self, project_id: str, key: str) -> AssetVersion | None:
        with self.uow:
            for asset in self.uow.assets.list_for_project(project_id):
                for version in self.uow.assets.list_versions(asset.asset_id):
                    if version.object_key == key:
                        return version
        return None


__all__ = ["MotionPipelineError", "MotionPipelineRecord", "MotionPipelineService", "MotionPipelineStore"]
