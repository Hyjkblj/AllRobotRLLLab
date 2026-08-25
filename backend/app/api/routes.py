"""HTTP API surface for P0/P1 contracts and P2 orchestration."""

from __future__ import annotations

import uuid
import json
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.adapters.motion import MotionDetectionError, MotionSourceRegistry
from backend.app.application.motion_editor import MotionArrays, MotionEditVersionStore, MotionEditor
from backend.app.application.asset_service import AssetService
from backend.app.application.artifact_service import ArtifactService
from backend.app.application.p3_dispatcher import P3DispatchError, P3DispatchService
from backend.app.application.reward_catalog import RewardConfigVersionStore, default_reward_catalog, validate_reward_config
from backend.app.application.run_service import RunService, RunServiceError, utc_now
from backend.app.application.training_validator import validate_training_config
from backend.app.application.training_service import TrainingService, TrainingServiceError
from backend.app.config.settings import settings
from backend.app.domain.contracts import Actor, ArtifactRecord, AssetKind, LicenseInfo, MotionEditConfig, ProjectRole, RewardConfig, RunManifest, RunStatus, Sim2SimThresholds, TrainingConfig
from backend.app.infrastructure.memory import InMemoryUnitOfWork
from backend.app.infrastructure.local import build_object_store
from backend.app.infrastructure.postgres import PostgresDatabase
from backend.app.infrastructure.postgres_uow import PostgresUnitOfWork
from backend.app.infrastructure.preflight import check_minio, check_redis
from backend.app.infrastructure.queue import CeleryTaskDispatcher, InMemoryTaskDispatcher


router = APIRouter()
g1_adapter = UnitreeG1Adapter(repository_root=settings.repository_root)
motion_registry = MotionSourceRegistry()
motion_edit_store = MotionEditVersionStore()
reward_config_store = RewardConfigVersionStore()
motion_editor = MotionEditor(g1_adapter.get_spec(), ik_solver=g1_adapter.create_ik_solver())
uow = PostgresUnitOfWork(settings.database_url) if settings.database_url else InMemoryUnitOfWork()
run_service = RunService(uow)
object_store = build_object_store(settings)
asset_service = AssetService(uow, object_store)
artifact_service = ArtifactService(uow, object_store)
training_service = TrainingService(run_service=run_service, robot_adapter=g1_adapter, workspace=settings.repository_root / ".runtime" / "p3", artifact_service=artifact_service, object_store=object_store)
if settings.execution_mode == "async" and settings.redis_url:
    p3_task_dispatcher = CeleryTaskDispatcher(settings.redis_url)
else:
    p3_task_dispatcher = InMemoryTaskDispatcher()
p3_dispatch_service = P3DispatchService(run_service=run_service, training_service=training_service, task_dispatcher=p3_task_dispatcher)


class MotionDetectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    asset_version_id: str | None = None


class MotionCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = Field(gt=0, le=120)
    joint_pos: list[list[float]] = Field(min_length=1)
    root_pos: list[list[float]] = Field(min_length=1)
    root_rot: list[list[float]] = Field(min_length=1)
    joint_names: list[str] = Field(min_length=1)
    quat_convention: str = "xyzw"
    coord_frame: str = "world_z_up"


class ProjectCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)


class ProjectMemberRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    role: ProjectRole


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str | None = None
    manifest: RunManifest | None = None
    robot: dict[str, str] | None = None
    motion: dict = Field(default_factory=dict)
    reward_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    runtime: dict[str, str] | None = None
    execution: dict = Field(default_factory=dict)
    licenses: list[LicenseInfo] = Field(default_factory=list)
    parent_run_id: str | None = None

    @classmethod
    def from_manifest(cls, manifest: RunManifest) -> "RunCreateRequest":
        return cls(project_id=manifest.project_id, manifest=manifest)


class RunStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    stage: str | None = None
    message: str = ""


class AssetCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: AssetKind
    display_name: str = Field(min_length=1, max_length=255)
    original_filename: str = Field(min_length=1, max_length=255)
    content_type: str | None = None
    license: LicenseInfo


class AssetUploadCompleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)


class AssetValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    rejection_code: str | None = None


class ArtifactRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    object_key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    content_type: str | None = None


class Sim2SimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seeds: list[int] = Field(default_factory=lambda: [20260101, 20260102, 20260103], min_length=3, max_length=3)
    thresholds: Sim2SimThresholds | None = None


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


def _actor(request: Request) -> Actor:
    user_id = request.headers.get("x-user-id", "local-user").strip() or "local-user"
    return Actor(user_id=user_id, email=request.headers.get("x-user-email"))


def _execution_mode(request: Request) -> str:
    return request.headers.get("x-execution-mode", settings.execution_mode).strip().lower()


def _error(request: Request, code: str, message: str, *, status_code: int = 400, details: dict | None = None) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"error": {"code": code, "message": message, "stage": "api", "details": details or {}, "retryable": False}, "request_id": _request_id(request)})


def _service_error(request: Request, error: RunServiceError) -> HTTPException:
    return _error(request, error.code, error.message, status_code=error.status_code)


@router.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "request_id": _request_id(request), "contract_versions": ["robot_spec.v1", "source_motion_descriptor.v1", "train_motion_npz.v1", "run_manifest.v1"]}


@router.get("/health/infrastructure")
def infrastructure_health(request: Request) -> dict:
    checks: dict[str, dict[str, object]] = {}
    if settings.database_url:
        try:
            checks["postgres"] = {"configured": True, "healthy": PostgresDatabase(settings.database_url).health()}
        except Exception as exc:
            checks["postgres"] = {"configured": True, "healthy": False, "error": type(exc).__name__}
    else:
        checks["postgres"] = {"configured": False, "healthy": False, "state": "pending"}
    if settings.redis_url:
        try:
            checks["redis"] = {"configured": True, "healthy": check_redis(settings.redis_url)}
        except Exception as exc:
            checks["redis"] = {"configured": True, "healthy": False, "error": type(exc).__name__}
    else:
        checks["redis"] = {"configured": False, "healthy": False, "state": "pending"}
    if settings.minio_endpoint and settings.minio_access_key and settings.minio_secret_key:
        try:
            checks["minio"] = {"configured": True, "healthy": check_minio(endpoint=settings.minio_endpoint, access_key=settings.minio_access_key, secret_key=settings.minio_secret_key, bucket=settings.minio_bucket, secure=settings.minio_secure)}
        except Exception as exc:
            checks["minio"] = {"configured": True, "healthy": False, "error": type(exc).__name__}
    else:
        checks["minio"] = {"configured": False, "healthy": False, "state": "pending"}
    status = "ok" if all(check["configured"] and check["healthy"] for check in checks.values()) else ("degraded" if any(check["configured"] and not check["healthy"] for check in checks.values()) else "pending")
    return {"status": status, "request_id": _request_id(request), "checks": checks}


@router.post("/projects")
def create_project(payload: ProjectCreateRequest, request: Request) -> dict:
    try:
        project = run_service.create_project(name=payload.name, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": project.model_dump(mode="json"), "resource_version": project.updated_at}


@router.get("/projects")
def list_projects(request: Request) -> dict:
    projects = run_service.list_projects(actor=_actor(request))
    return {"request_id": _request_id(request), "items": [project.model_dump(mode="json") for project in projects], "resource_version": str(len(projects))}


@router.get("/projects/{project_id}")
def get_project(project_id: str, request: Request) -> dict:
    try:
        project, members = run_service.get_project(project_id=project_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": project.model_dump(mode="json"), "members": [member.model_dump(mode="json") for member in members], "resource_version": project.updated_at}


@router.post("/projects/{project_id}/members")
def add_project_member(project_id: str, payload: ProjectMemberRequest, request: Request) -> dict:
    try:
        member = run_service.add_member(project_id=project_id, actor=_actor(request), user_id=payload.user_id, role=payload.role)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": member.model_dump(mode="json"), "resource_version": member.created_at}


@router.post("/projects/{project_id}/assets")
def create_asset(project_id: str, payload: AssetCreateRequest, request: Request) -> dict:
    try:
        asset, version, session = asset_service.create_asset(actor=_actor(request), project_id=project_id, kind=payload.kind, display_name=payload.display_name, original_filename=payload.original_filename, license=payload.license, content_type=payload.content_type)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": asset.model_dump(mode="json"), "version": version.model_dump(mode="json"), "upload": session.model_dump(mode="json"), "resource_version": version.asset_version_id}


@router.get("/assets/{asset_id}/versions")
def list_asset_versions(asset_id: str, request: Request) -> dict:
    try:
        asset, versions = asset_service.list_versions(actor=_actor(request), asset_id=asset_id)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "asset": asset.model_dump(mode="json"), "items": [version.model_dump(mode="json") for version in versions], "resource_version": str(len(versions))}


@router.post("/assets/{asset_version_id}/upload-complete")
def complete_asset_upload(asset_version_id: str, payload: AssetUploadCompleteRequest, request: Request) -> dict:
    try:
        version = asset_service.complete_upload(actor=_actor(request), asset_version_id=asset_version_id, sha256=payload.sha256, size_bytes=payload.size_bytes)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": version.model_dump(mode="json"), "resource_version": version.asset_version_id}


@router.post("/assets/{asset_version_id}/upload-url")
def asset_upload_url(asset_version_id: str, request: Request) -> dict:
    try:
        session = asset_service.upload_session(actor=_actor(request), asset_version_id=asset_version_id)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "upload": session.model_dump(mode="json"), "resource_version": session.asset_version_id}


@router.post("/assets/{asset_version_id}/validate")
def validate_asset(asset_version_id: str, payload: AssetValidateRequest, request: Request) -> dict:
    # This endpoint is intended for an authenticated internal worker.  The
    # worker token integration is added at deployment; local P2 tests use the
    # explicit X-Worker-Id marker to avoid exposing it as a user action.
    if not request.headers.get("x-worker-id"):
        raise _error(request, "WORKER_AUTH_REQUIRED", "asset validation is an internal worker operation", status_code=403)
    try:
        version = asset_service.mark_validated(asset_version_id=asset_version_id, valid=payload.valid, sha256=payload.sha256, size_bytes=payload.size_bytes, rejection_code=payload.rejection_code)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": version.model_dump(mode="json"), "resource_version": version.asset_version_id}


@router.post("/artifacts")
def register_artifact(payload: ArtifactRegisterRequest, request: Request) -> dict:
    if not request.headers.get("x-worker-id"):
        raise _error(request, "WORKER_AUTH_REQUIRED", "artifact registration is an internal worker operation", status_code=403)
    artifact = ArtifactRecord(artifact_id=str(uuid.uuid4()), created_at=utc_now(), **payload.model_dump())
    # Keep the run/attempt relationship authoritative in PostgreSQL/in-memory
    # repositories before accepting a worker-produced object index.
    with uow:
        run = uow.runs.get(payload.run_id)
        attempt = uow.runs.attempt(payload.attempt_id)
    if run is None or attempt is None or attempt.run_id != run.run_id:
        raise _error(request, "ARTIFACT_ATTEMPT_INVALID", "artifact run/attempt relationship is invalid")
    try:
        record = artifact_service.register(artifact)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": record.model_dump(mode="json"), "resource_version": record.sha256}


@router.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str, request: Request) -> dict:
    try:
        artifact, url = artifact_service.get_download(artifact_id=artifact_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": artifact.model_dump(mode="json"), "download_url": url, "resource_version": artifact.sha256}


@router.get("/robots")
def robots(request: Request) -> dict:
    spec = g1_adapter.get_spec()
    return {"request_id": _request_id(request), "items": [spec.model_dump(mode="json")], "resource_version": spec.adapter_version}


@router.get("/robots/{robot_id}")
def robot(robot_id: str, request: Request) -> dict:
    if robot_id != g1_adapter.name:
        raise _error(request, "ROBOT_NOT_FOUND", f"unknown robot adapter: {robot_id}", status_code=404)
    spec = g1_adapter.get_spec()
    return {"request_id": _request_id(request), "item": spec.model_dump(mode="json"), "resource_version": spec.adapter_version}


@router.get("/robots/{robot_id}/self-check")
def robot_self_check(robot_id: str, request: Request) -> dict:
    if robot_id != g1_adapter.name:
        raise _error(request, "ROBOT_NOT_FOUND", f"unknown robot adapter: {robot_id}", status_code=404)
    result = g1_adapter.self_check()
    return {"request_id": _request_id(request), "result": result.model_dump(mode="json"), "resource_version": g1_adapter.get_spec().adapter_version}


@router.get("/reward-templates")
def reward_templates(request: Request) -> dict:
    items = [term.model_dump(mode="json") for term in default_reward_catalog()]
    return {"request_id": _request_id(request), "items": items, "resource_version": "reward-registry.v1"}


@router.get("/training-config/schema")
def training_config_schema(request: Request) -> dict:
    return {"request_id": _request_id(request), "schema": TrainingConfig.model_json_schema(), "resource_version": "training_config.v1"}


@router.post("/motions/detect")
def detect_motion(payload: MotionDetectRequest, request: Request) -> dict:
    path = Path(payload.path).expanduser().resolve()
    try:
        path.relative_to(settings.motion_asset_root)
    except ValueError as exc:
        raise _error(request, "INPUT_PATH_NOT_ALLOWED", "motion path is outside the configured asset root", details={"asset_root": str(settings.motion_asset_root)}) from exc
    try:
        # Trusted pickle parsing is an internal worker capability, never a
        # user-controlled API flag.
        descriptor = motion_registry.detect(path, asset_version_id=payload.asset_version_id, trusted_pickle=False)
    except MotionDetectionError as exc:
        raise _error(request, exc.code, exc.message, details=exc.details) from exc
    return {"request_id": _request_id(request), "descriptor": descriptor.model_dump(mode="json"), "resource_version": descriptor.detector_version}


@router.post("/motion-edits")
def create_motion_edit(payload: MotionEditConfig, request: Request, parent_version_id: str | None = Query(default=None)) -> dict:
    if payload.robot_id != g1_adapter.name:
        raise _error(request, "ROBOT_NOT_FOUND", f"unknown robot adapter: {payload.robot_id}", status_code=404)
    try:
        record = motion_edit_store.create(payload, parent_version_id=parent_version_id)
    except ValueError as exc:
        raise _error(request, "MOTION_EDIT_VERSION_INVALID", str(exc)) from exc
    return {"request_id": _request_id(request), "item": record.model_dump(mode="json"), "resource_version": record.config_sha256}


@router.get("/motion-edits/{version_id}")
def get_motion_edit(version_id: str, request: Request) -> dict:
    record = motion_edit_store.get(version_id)
    if record is None:
        raise _error(request, "MOTION_EDIT_NOT_FOUND", f"motion edit version not found: {version_id}", status_code=404)
    return {"request_id": _request_id(request), "item": record.model_dump(mode="json"), "resource_version": record.config_sha256}


@router.get("/motions/{source_motion_version_id}/edits")
def list_motion_edits(source_motion_version_id: str, request: Request) -> dict:
    records = motion_edit_store.list_for_source(source_motion_version_id)
    return {"request_id": _request_id(request), "items": [record.model_dump(mode="json") for record in records], "resource_version": str(len(records))}


@router.post("/motion-edits/{version_id}/compile")
def compile_motion_edit(version_id: str, payload: MotionCompileRequest, request: Request) -> dict:
    record = motion_edit_store.get(version_id)
    if record is None:
        raise _error(request, "MOTION_EDIT_NOT_FOUND", f"motion edit version not found: {version_id}", status_code=404)
    arrays = MotionArrays(
        fps=payload.fps,
        joint_pos=np.asarray(payload.joint_pos, dtype=np.float64),
        root_pos=np.asarray(payload.root_pos, dtype=np.float64),
        root_rot=np.asarray(payload.root_rot, dtype=np.float64),
        joint_names=tuple(payload.joint_names),
        quat_convention=payload.quat_convention,
        coord_frame=payload.coord_frame,
    )
    result = motion_editor.apply(arrays, record.config)
    response = {
        "request_id": _request_id(request),
        "result": result.validation.model_dump(mode="json"),
        "quality": result.quality.model_dump(mode="json"),
        "resource_version": record.config_sha256,
    }
    if result.arrays is not None:
        response["motion"] = {"fps": result.arrays.fps, "frame_count": len(result.arrays.joint_pos), "joint_names": list(result.arrays.joint_names), "shapes": {"joint_pos": list(result.arrays.joint_pos.shape), "root_pos": list(result.arrays.root_pos.shape), "root_rot": list(result.arrays.root_rot.shape)}}
    return response


@router.post("/motion-edits/{version_id}/restore/{target_version_id}")
def restore_motion_edit(version_id: str, target_version_id: str, request: Request) -> dict:
    try:
        record = motion_edit_store.restore(version_id, target_version_id)
    except ValueError as exc:
        raise _error(request, "MOTION_EDIT_RESTORE_INVALID", str(exc)) from exc
    return {"request_id": _request_id(request), "item": record.model_dump(mode="json"), "resource_version": record.config_sha256}


@router.post("/reward-configs/validate")
def validate_reward(payload: RewardConfig, request: Request) -> dict:
    result = validate_reward_config(payload)
    return {"request_id": _request_id(request), "result": result.model_dump(mode="json"), "resource_version": result.processor_version}


@router.post("/reward-configs")
def create_reward_config(payload: RewardConfig, request: Request, parent_version_id: str | None = Query(default=None)) -> dict:
    try:
        record = reward_config_store.create(payload, parent_version_id=parent_version_id)
    except ValueError as exc:
        raise _error(request, "REWARD_CONFIG_INVALID", str(exc)) from exc
    return {"request_id": _request_id(request), "item": record.model_dump(mode="json"), "resource_version": record.config_sha256}


@router.get("/reward-configs/{version_id}")
def get_reward_config(version_id: str, request: Request) -> dict:
    record = reward_config_store.get(version_id)
    if record is None:
        raise _error(request, "REWARD_CONFIG_NOT_FOUND", f"reward config version not found: {version_id}", status_code=404)
    return {"request_id": _request_id(request), "item": record.model_dump(mode="json"), "resource_version": record.config_sha256}


@router.get("/reward-templates/{template}/versions")
def list_reward_versions(template: str, request: Request) -> dict:
    records = reward_config_store.list_for_template(template)
    return {"request_id": _request_id(request), "items": [record.model_dump(mode="json") for record in records], "resource_version": str(len(records))}


@router.post("/training-config/validate")
def validate_training(payload: TrainingConfig, request: Request) -> dict:
    result = validate_training_config(payload, g1_adapter.get_spec())
    return {"request_id": _request_id(request), "result": result.model_dump(mode="json"), "resource_version": result.processor_version}


def _create_run_arguments(payload: RunCreateRequest, request: Request) -> dict:
    if payload.manifest is not None:
        manifest = payload.manifest
        return {
            "project_id": manifest.project_id,
            "robot": manifest.robot,
            "motion": manifest.motion,
            "reward_config_sha256": manifest.reward_config_sha256,
            "training_config_sha256": manifest.training_config_sha256,
            "runtime": manifest.runtime.model_dump(mode="json"),
            "execution": manifest.execution,
            "licenses": manifest.licenses,
            "parent_run_id": manifest.parent_run_id,
        }
    if not payload.project_id or not payload.robot or not payload.reward_config_sha256 or not payload.training_config_sha256:
        raise _error(request, "RUN_REQUEST_INVALID", "project_id, robot, motion, reward and training hashes are required")
    return {
        "project_id": payload.project_id,
        "robot": payload.robot,
        "motion": payload.motion,
        "reward_config_sha256": payload.reward_config_sha256,
        "training_config_sha256": payload.training_config_sha256,
        "runtime": payload.runtime,
        "execution": payload.execution,
        "licenses": payload.licenses,
        "parent_run_id": payload.parent_run_id,
    }


@router.post("/runs")
def create_run(payload: RunCreateRequest, request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict:
    try:
        values = _create_run_arguments(payload, request)
        run, attempt, replayed = run_service.create_run(actor=_actor(request), idempotency_key=idempotency_key, **values)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": run.model_dump(mode="json"), "attempt": attempt.model_dump(mode="json"), "replayed": replayed, "resource_version": run.manifest.manifest_sha256}


@router.get("/runs/{run_id}/events")
def run_events(run_id: str, request: Request, after_seq: int = Query(default=0, ge=0), attempt_id: str | None = Query(default=None), last_event_id: int | None = Header(default=None, alias="Last-Event-ID")) -> StreamingResponse:
    cursor = last_event_id if last_event_id is not None else after_seq
    try:
        events = run_service.list_events(run_id=run_id, actor=_actor(request), after_seq=cursor, attempt_id=attempt_id)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc

    def stream():
        for event in events:
            yield f"id: {event.seq}\nevent: {event.event_type}\ndata: {json.dumps(event.model_dump(mode='json'), ensure_ascii=False, separators=(',', ':'))}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    try:
        run, attempts = run_service.get_run(run_id=run_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": run.model_dump(mode="json"), "attempts": [attempt.model_dump(mode="json") for attempt in attempts], "resource_version": run.updated_at}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request) -> dict:
    try:
        run, attempt = run_service.cancel_run(run_id=run_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": run.model_dump(mode="json"), "attempt": attempt.model_dump(mode="json"), "resource_version": run.updated_at}


@router.post("/runs/{run_id}/retry")
def retry_run(run_id: str, request: Request) -> dict:
    try:
        run, attempt = run_service.retry_run(run_id=run_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": run.model_dump(mode="json"), "attempt": attempt.model_dump(mode="json"), "resource_version": run.updated_at}


@router.post("/runs/{run_id}/status")
def update_run_status(run_id: str, payload: RunStatusRequest, request: Request) -> dict:
    worker_id = request.headers.get("x-worker-id")
    if not worker_id:
        raise _error(request, "WORKER_AUTH_REQUIRED", "run status updates are an internal worker operation", status_code=403)
    try:
        run = run_service.transition_run(run_id=run_id, target=payload.status, stage=payload.stage, message=payload.message)
        run, attempts = run_service.get_run(run_id=run_id, actor=Actor(user_id=run.created_by))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": run.model_dump(mode="json"), "attempts": [attempt.model_dump(mode="json") for attempt in attempts], "resource_version": run.updated_at}


class RunHeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_uuid: str | None = None


class RunEventAppendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["status", "log", "metric", "system"]
    stage: str | None = None
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    message: str = ""
    payload: dict = Field(default_factory=dict)


@router.post("/runs/{run_id}/heartbeat")
def heartbeat_run(run_id: str, payload: RunHeartbeatRequest, request: Request) -> dict:
    worker_id = request.headers.get("x-worker-id")
    if not worker_id:
        raise _error(request, "WORKER_AUTH_REQUIRED", "run heartbeat is an internal worker operation", status_code=403)
    try:
        attempt = run_service.heartbeat(run_id=run_id, worker_id=worker_id, gpu_uuid=payload.gpu_uuid)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "attempt": attempt.model_dump(mode="json"), "resource_version": attempt.last_heartbeat_at}


@router.post("/runs/{run_id}/events")
def append_run_event(run_id: str, payload: RunEventAppendRequest, request: Request) -> dict:
    if not request.headers.get("x-worker-id"):
        raise _error(request, "WORKER_AUTH_REQUIRED", "run event writes are an internal worker operation", status_code=403)
    try:
        event = run_service.append_event(run_id=run_id, event_type=payload.event_type, stage=payload.stage, level=payload.level, message=payload.message, payload=payload.payload)
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "item": event.model_dump(mode="json"), "resource_version": str(event.seq)}


@router.post("/runs/{run_id}/train")
def train_run(run_id: str, payload: TrainingConfig, request: Request) -> dict:
    if not request.headers.get("x-worker-id"):
        raise _error(request, "WORKER_AUTH_REQUIRED", "training is an internal worker operation", status_code=403)
    if _execution_mode(request) == "async":
        try:
            submission = p3_dispatch_service.submit_train(run_id=run_id, config=payload, actor=_actor(request), worker_id=request.headers["x-worker-id"])
        except P3DispatchError as exc:
            raise _error(request, exc.code, exc.message, status_code=exc.status_code) from exc
        return JSONResponse(status_code=202, content={"request_id": _request_id(request), "submission": submission.model_dump(mode="json"), "resource_version": submission.idempotency_key})
    try:
        result = training_service.train_smoke(run_id=run_id, config=payload, worker_id=request.headers["x-worker-id"])
    except TrainingServiceError as exc:
        raise _error(request, exc.code, exc.message, status_code=exc.status_code) from exc
    return {"request_id": _request_id(request), "checkpoint": result.checkpoint.model_dump(mode="json"), "metrics": [metric.model_dump(mode="json") for metric in result.metrics], "artifacts": [artifact.model_dump(mode="json") for artifact in result.artifacts], "resource_version": result.checkpoint.sha256}


@router.post("/runs/{run_id}/export")
def export_run(run_id: str, request: Request) -> dict:
    if not request.headers.get("x-worker-id"):
        raise _error(request, "WORKER_AUTH_REQUIRED", "export is an internal worker operation", status_code=403)
    if _execution_mode(request) == "async":
        try:
            submission = p3_dispatch_service.submit_export(run_id=run_id, actor=_actor(request), worker_id=request.headers["x-worker-id"])
        except P3DispatchError as exc:
            raise _error(request, exc.code, exc.message, status_code=exc.status_code) from exc
        return JSONResponse(status_code=202, content={"request_id": _request_id(request), "submission": submission.model_dump(mode="json"), "resource_version": submission.idempotency_key})
    try:
        bundle = training_service.export(run_id=run_id)
    except TrainingServiceError as exc:
        raise _error(request, exc.code, exc.message, status_code=exc.status_code) from exc
    return {"request_id": _request_id(request), "item": bundle.model_dump(mode="json"), "resource_version": bundle.bundle_id}


@router.post("/runs/{run_id}/sim2sim")
def sim2sim_run(run_id: str, payload: Sim2SimRequest, request: Request) -> dict:
    if not request.headers.get("x-worker-id"):
        raise _error(request, "WORKER_AUTH_REQUIRED", "sim2sim is an internal worker operation", status_code=403)
    if _execution_mode(request) == "async":
        try:
            submission = p3_dispatch_service.submit_sim2sim(run_id=run_id, seeds=tuple(payload.seeds), thresholds=payload.thresholds, actor=_actor(request), worker_id=request.headers["x-worker-id"])
        except P3DispatchError as exc:
            raise _error(request, exc.code, exc.message, status_code=exc.status_code) from exc
        return JSONResponse(status_code=202, content={"request_id": _request_id(request), "submission": submission.model_dump(mode="json"), "resource_version": submission.idempotency_key})
    try:
        report = training_service.sim2sim(run_id=run_id, seeds=tuple(payload.seeds), thresholds=payload.thresholds)
    except TrainingServiceError as exc:
        raise _error(request, exc.code, exc.message, status_code=exc.status_code) from exc
    return {"request_id": _request_id(request), "item": report.model_dump(mode="json"), "resource_version": report.report_sha256}


@router.get("/runs/{run_id}/sim2sim")
def get_sim2sim_report(run_id: str, request: Request) -> dict:
    try:
        run, _ = run_service.get_run(run_id=run_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    report = training_service.reports.get(run_id)
    if report is None:
        raise _error(request, "SIM2SIM_REPORT_NOT_FOUND", f"sim2sim report not found for run: {run_id}", status_code=404)
    return {"request_id": _request_id(request), "item": report.model_dump(mode="json"), "resource_version": report.report_sha256}


@router.get("/runs/{run_id}/artifacts")
def list_run_artifacts(run_id: str, request: Request) -> dict:
    try:
        artifacts = artifact_service.list_for_run(run_id=run_id, actor=_actor(request))
    except RunServiceError as exc:
        raise _service_error(request, exc) from exc
    return {"request_id": _request_id(request), "items": [artifact.model_dump(mode="json") for artifact in artifacts], "resource_version": str(len(artifacts))}
