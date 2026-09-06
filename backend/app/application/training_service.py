"""P3 training/play/export orchestration with a deterministic CPU smoke backend."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.application.policy_exporter import ExportError, ExportResult, TorchPolicyExporter, build_policy_bundle
from backend.app.application.run_service import RunService, RunServiceError, utc_now
from backend.app.application.sim2sim_service import FakeSim2SimAdapter, build_sim2sim_report
from backend.app.application.training_validator import validate_training_config
from backend.app.application.robot_catalog import RobotAdapterRegistry, RobotRegistryError
from backend.app.application.task_catalog import TaskRegistry, TaskRegistryError, default_task_registry
from backend.app.application.provider_catalog import Sim2SimRegistry, TrainingProviderRegistry
from backend.app.domain.contracts import Actor, ArtifactRecord, CheckpointRecord, ExportMetadata, MetricPoint, P3RunState, PolicyBundle, RobotSpec, Sim2SimReport, Sim2SimThresholds, TrainingConfig
from backend.app.domain.contracts import AssetKind, AssetVersionStatus
from backend.app.domain.state_machine import RunStatus
from backend.app.runtime.contracts import RunnerError
from backend.app.runtime.process import external_run_context
from backend.app.config.settings import settings
from backend.app.application.policy_exporter import file_record


class TrainingServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: CheckpointRecord
    metrics: list[MetricPoint]
    output_dir: Path
    artifacts: list[ArtifactRecord] = field(default_factory=list)


class TrainingService:
    def __init__(self, *, run_service: RunService, robot_adapter=None, workspace: Path, artifact_service=None, object_store=None, training_runner=None, sim2sim_adapter=None, robot_registry: RobotAdapterRegistry | None = None, task_registry: TaskRegistry | None = None, training_providers: dict[str, object] | None = None, sim2sim_adapters: dict[str, object] | None = None, training_provider_registry: TrainingProviderRegistry | None = None, sim2sim_registry: Sim2SimRegistry | None = None, reward_config_store=None) -> None:
        self.run_service = run_service
        self.robot_adapter = robot_adapter
        if robot_registry is None:
            if robot_adapter is None:
                raise ValueError("robot_registry or robot_adapter is required")
            robot_registry = RobotAdapterRegistry([robot_adapter])
        self.robot_registry = robot_registry
        # Compatibility callers historically supplied only a concrete adapter.
        # Derive the task catalog from that adapter instead of silently using
        # an empty registry; composition roots still pass their explicit
        # multi-robot registry here.
        self.task_registry = task_registry or default_task_registry(robot_registry)
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifact_service = artifact_service
        self.object_store = object_store
        self.training_runner = training_runner
        self.sim2sim_adapter = sim2sim_adapter
        self.reward_config_store = reward_config_store
        self.training_providers = dict(training_providers or {})
        if training_runner is not None:
            self.training_providers.setdefault(getattr(training_runner, "name", "default"), training_runner)
        self.sim2sim_adapters = dict(sim2sim_adapters or {})
        if sim2sim_adapter is not None:
            self.sim2sim_adapters.setdefault(getattr(sim2sim_adapter, "name", "default"), sim2sim_adapter)
        # An explicitly supplied provider map is authoritative. This allows
        # fake_smoke workers to receive an empty map even when runtime probes
        # have discovered external providers on the same host.
        self.training_provider_registry = (
            TrainingProviderRegistry(self.training_providers.values())
            if training_providers is not None or training_runner is not None
            else (training_provider_registry or TrainingProviderRegistry())
        )
        self.sim2sim_registry = (
            Sim2SimRegistry(self.sim2sim_adapters.values())
            if sim2sim_adapters is not None or sim2sim_adapter is not None
            else (sim2sim_registry or Sim2SimRegistry())
        )
        self.configs: dict[str, TrainingConfig] = {}
        self.checkpoints: dict[str, CheckpointRecord] = {}
        self.bundles: dict[str, PolicyBundle] = {}
        self.reports: dict[str, Sim2SimReport] = {}
        self.export_results: dict[str, ExportResult] = {}
        self.bundle_dirs: dict[str, Path] = {}
        self.artifact_ids: dict[str, list[str]] = {}
        self.reward_configs: dict[str, object] = {}

    def prepare_training(self, *, run_id: str, config: TrainingConfig, actor: Actor | None = None):
        """Validate and reserve a run for a training worker.

        This method intentionally stops at ``TRAINING_PREPARING`` so an
        asynchronous dispatcher can enqueue the job before GPU work begins.
        The synchronous smoke path advances the final step itself.
        """
        effective_actor = actor or Actor(user_id=self._creator(run_id))
        run, _attempts = self.run_service.get_run(run_id=run_id, actor=effective_actor)
        adapter = self._adapter_for_run(run)
        config = self._resolve_config_for_run(run, config, adapter=adapter)
        try:
            task = self.task_registry.get(config.task_id)
        except TaskRegistryError as exc:
            raise TrainingServiceError("TASK_NOT_FOUND", str(exc), status_code=422) from exc
        validation = validate_training_config(config, adapter.get_spec(), task)
        if not validation.valid:
            raise TrainingServiceError("TRAIN_CONFIG_INVALID", validation.model_dump_json())
        reward_config = self._resolve_reward_config(run, robot_id=adapter.name, task_id=task.task_id)
        # When the asset repository is available, refuse source/processing
        # versions that have not completed the motion pipeline.  Contract
        # tests may use synthetic motion ids, so an absent record remains
        # valid for the isolated smoke service.
        with self.run_service.uow:
            motion_version = self.run_service.uow.assets.version(config.motion_asset_version_id)
            motion_asset = self.run_service.uow.assets.get(motion_version.asset_id) if motion_version else None
        if motion_version is not None:
            if motion_asset is None or motion_asset.kind != AssetKind.MOTION:
                raise TrainingServiceError("TRAIN_MOTION_INVALID", "training input is not a motion asset", status_code=409)
            if motion_asset.project_id != run.project_id:
                raise TrainingServiceError("TRAIN_MOTION_PROJECT_MISMATCH", "training input belongs to a different project", status_code=403)
            if motion_version.status != AssetVersionStatus.READY:
                raise TrainingServiceError("TRAIN_MOTION_NOT_READY", f"training input must be READY, got {motion_version.status}", status_code=409)
            if "train_motion" not in (motion_version.original_filename or "").lower() and "trainmotionnpz" not in (motion_asset.display_name or "").lower():
                raise TrainingServiceError("TRAIN_MOTION_NOT_COMPILED", "training input must be a compiled TrainMotionNPZ asset", status_code=409)
        self.configs[run_id] = config
        if reward_config is not None:
            self.reward_configs[run_id] = reward_config
        if run.status == RunStatus.CREATED:
            for status in (RunStatus.VALIDATING, RunStatus.MOTION_COMPILING, RunStatus.MOTION_READY, RunStatus.TRAINING_PREPARING):
                self.run_service.transition_run(run_id=run_id, target=status, stage=status.value.lower(), message=f"P3 training preparation: {status.value}")
        elif run.status not in (RunStatus.TRAINING_PREPARING, RunStatus.TRAINING):
            raise TrainingServiceError("TRAIN_STATUS_INVALID", f"run must be CREATED, TRAINING_PREPARING or TRAINING, got {run.status}", status_code=409)
        run, attempts = self.run_service.get_run(run_id=run_id, actor=effective_actor)
        attempt = next(attempt for attempt in attempts if attempt.attempt_id == run.current_attempt_id)
        state = self._load_state(run_id)
        self._persist_state(state.model_copy(update={"attempt_id": attempt.attempt_id, "training_config": config, "updated_at": utc_now()}))
        return run, attempt

    def train_smoke(self, *, run_id: str, config: TrainingConfig, worker_id: str = "local-smoke-worker") -> TrainingResult:
        run, attempt = self.prepare_training(run_id=run_id, config=config)
        config = self.configs.get(run_id, config)
        if run.status == RunStatus.TRAINING_PREPARING:
            self.run_service.transition_run(run_id=run_id, target=RunStatus.TRAINING, stage="training", message="P3 smoke training started")
        output_dir = self.workspace / "runs" / run_id / attempt.attempt_id
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter = self._adapter_for_run(run)
        robot = adapter.get_spec()
        obs_dim = self.observation_dim(config, robot)
        action_dim = robot.dof
        checkpoint_payload = {"format": "robot_mimic_checkpoint.v1", "run_id": run_id, "attempt_id": attempt.attempt_id, "robot_id": adapter.name, "task_id": config.task_id, "observation_dim": obs_dim, "action_dim": action_dim, "iteration": config.ppo.max_iterations, "worker_id": worker_id}
        checkpoint_path = output_dir / "checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_payload, sort_keys=True, indent=2), encoding="utf-8")
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = CheckpointRecord(checkpoint_id=str(uuid.uuid4()), run_id=run_id, attempt_id=attempt.attempt_id, uri=str(checkpoint_path), sha256=checkpoint_hash, iteration=config.ppo.max_iterations, created_at=utc_now())
        self.checkpoints[run_id] = checkpoint
        artifacts: list[ArtifactRecord] = []
        checkpoint_artifact = self._register_file_artifact(run_id=run_id, attempt_id=attempt.attempt_id, kind="checkpoint", path=checkpoint_path, content_type="application/json")
        if checkpoint_artifact is not None:
            artifacts.append(checkpoint_artifact)
        state = self._load_state(run_id)
        self.artifact_ids[run_id] = [artifact.artifact_id for artifact in artifacts]
        self._persist_state(state.model_copy(update={"attempt_id": attempt.attempt_id, "training_config": config, "checkpoint": checkpoint, "artifact_ids": list(self.artifact_ids[run_id]), "updated_at": utc_now()}))
        metrics = [MetricPoint(attempt_id=attempt.attempt_id, step=step, name=name, value=value, timestamp=utc_now()) for step, (name, value) in enumerate((("train/return", 0.0), ("train/return", 1.0), ("train/fall_rate", 0.0), ("train/joint_rmse_rad", 0.1), ("resource/gpu_memory_gb", float(config.resources.gpu_memory_gb))))]
        for metric in metrics:
            self.run_service.append_event(run_id=run_id, event_type="metric", stage="training", message=metric.name, payload=metric.model_dump(mode="json"))
        self._ensure_not_cancelled(run_id)
        self.run_service.transition_run(run_id=run_id, target=RunStatus.TRAINING_SUCCEEDED, stage="training", message="P3 smoke training completed")
        return TrainingResult(checkpoint=checkpoint, metrics=metrics, output_dir=output_dir, artifacts=artifacts)

    def train(self, *, run_id: str, config: TrainingConfig, worker_id: str = "worker") -> TrainingResult:
        """Run the configured Isaac backend, or the deterministic dev backend."""
        # Resolve the Run first so a globally selected provider cannot execute
        # a task for a different robot (for example H1 through the G1 provider).
        actor = Actor(user_id=self._creator(run_id))
        run, _ = self.run_service.get_run(run_id=run_id, actor=actor)
        selected_adapter = self._adapter_for_run(run)
        effective_config = self._resolve_config_for_run(run, config, adapter=selected_adapter)
        provider = self._provider_for_task(effective_config.task_id, robot_id=selected_adapter.name)
        if provider is None:
            return self.train_smoke(run_id=run_id, config=config, worker_id=worker_id)
        run, attempt = self.prepare_training(run_id=run_id, config=effective_config)
        config = self.configs.get(run_id, effective_config)
        if run.status == RunStatus.TRAINING_PREPARING:
            self.run_service.transition_run(run_id=run_id, target=RunStatus.TRAINING, stage="training", message="Isaac Lab training started")
        output_dir = self.workspace / "runs" / run_id / attempt.attempt_id
        try:
            motion_path = self._materialize_asset(config.motion_asset_version_id, output_dir / "input" / "train_motion.npz")
            provider_config = config.model_dump(mode="json")
            reward_config = self.reward_configs.get(run_id)
            if reward_config is not None:
                provider_config["reward_config"] = reward_config.model_dump(mode="json")
                provider_config["reward_config_sha256"] = run.manifest.reward_config_sha256
            # The selected provider receives the frozen Run Manifest through a
            # private transport key. Provider implementations may persist it
            # as their own input contract without coupling the domain model to
            # an external process API.
            provider_config["_run_manifest"] = run.manifest.model_dump(mode="json")
            with external_run_context(run_id=run_id, runtime_root=settings.runtime_root):
                execution = provider.train(run_id=run_id, task_id=config.task_id, motion_path=motion_path, config=provider_config, output_dir=output_dir)
            self._ensure_not_cancelled(run_id)
            checkpoint_path = execution.checkpoint_path
            checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            checkpoint = CheckpointRecord(checkpoint_id=str(uuid.uuid4()), run_id=run_id, attempt_id=attempt.attempt_id, uri=str(checkpoint_path), sha256=checkpoint_hash, iteration=execution.iteration, created_at=utc_now())
            self.checkpoints[run_id] = checkpoint
            artifacts: list[ArtifactRecord] = []
            artifact = self._register_file_artifact(run_id=run_id, attempt_id=attempt.attempt_id, kind="checkpoint", path=checkpoint_path, content_type="application/octet-stream")
            if artifact is not None:
                artifacts.append(artifact)
            metrics = [MetricPoint(attempt_id=attempt.attempt_id, step=int(item.get("iteration", item.get("step", index))), name=str(item.get("name", "train/return")), value=float(item.get("value", item.get("train/return", 0.0))), timestamp=utc_now()) for index, item in enumerate(execution.metrics)]
            for metric in metrics:
                self.run_service.append_event(run_id=run_id, event_type="metric", stage="training", message=metric.name, payload=metric.model_dump(mode="json"))
            self.artifact_ids[run_id] = [item.artifact_id for item in artifacts]
            state = self._load_state(run_id)
            self._persist_state(state.model_copy(update={"attempt_id": attempt.attempt_id, "training_config": config, "checkpoint": checkpoint, "artifact_ids": list(self.artifact_ids[run_id]), "updated_at": utc_now()}))
            self.run_service.transition_run(run_id=run_id, target=RunStatus.TRAINING_SUCCEEDED, stage="training", message="Isaac Lab training completed")
            return TrainingResult(checkpoint=checkpoint, metrics=metrics, output_dir=output_dir, artifacts=artifacts)
        except RunnerError as exc:
            self._mark_failed_if_active(run_id, stage="training", message=str(exc))
            raise TrainingServiceError(exc.code, str(exc), status_code=503) from exc

    def _materialize_asset(self, asset_version_id: str, destination: Path) -> Path:
        with self.run_service.uow:
            version = self.run_service.uow.assets.version(asset_version_id)
        if version is None:
            raise TrainingServiceError("TRAIN_MOTION_NOT_FOUND", f"motion asset version not found: {asset_version_id}", status_code=404)
        resolver = getattr(self.object_store, "resolve_path", None)
        if callable(resolver):
            try:
                source = resolver(version.object_key)
                if source.is_file():
                    return source
            except Exception:
                pass
        downloader = getattr(self.object_store, "download_file", None)
        if not callable(downloader):
            raise TrainingServiceError("TRAIN_MOTION_UNAVAILABLE", "object store cannot materialize training motion", status_code=503)
        try:
            return downloader(version.object_key, destination)
        except Exception as exc:
            raise TrainingServiceError("TRAIN_MOTION_UNAVAILABLE", f"unable to download training motion: {exc}", status_code=503) from exc

    def export(self, *, run_id: str, exporter=None) -> PolicyBundle:
        run, _attempts = self.run_service.get_run(run_id=run_id, actor=Actor(user_id=self._creator(run_id)))
        adapter = self._adapter_for_run(run)
        robot = adapter.get_spec()
        if run.status != RunStatus.TRAINING_SUCCEEDED:
            raise TrainingServiceError("EXPORT_STATUS_INVALID", f"run must be TRAINING_SUCCEEDED, got {run.status}", status_code=409)
        state = self._load_state(run_id)
        config = state.training_config or self.configs.get(run_id)
        checkpoint = state.checkpoint or self.checkpoints.get(run_id)
        if config is None or checkpoint is None:
            raise TrainingServiceError("CHECKPOINT_NOT_FOUND", "training checkpoint and config are required before export", status_code=409)
        self.run_service.transition_run(run_id=run_id, target=RunStatus.EXPORTING, stage="export", message="Export started")
        output_dir = Path(checkpoint.uri).parent / "export"
        export_impl = exporter
        try:
            provider = self._provider_for_task(config.task_id, robot_id=adapter.name)
            if export_impl is None and provider is not None and hasattr(provider, "export"):
                with external_run_context(run_id=run_id, runtime_root=settings.runtime_root):
                    execution = provider.export(checkpoint_path=Path(checkpoint.uri), task_id=config.task_id, output_dir=output_dir)
                self._ensure_not_cancelled(run_id)
                exported = self._external_export_result(execution, output_dir=output_dir, input_dim=self.observation_dim(config, robot), output_dim=robot.dof, action_scale=robot.actuation.action_scale, provider_name=getattr(provider, "name", "external"))
            else:
                export_impl = export_impl or TorchPolicyExporter()
                exported = export_impl.export(output_dir=output_dir, input_dim=self.observation_dim(config, robot), output_dim=robot.dof, action_scale=robot.actuation.action_scale)
        except RunnerError as exc:
            self._mark_failed_if_active(run_id, stage="export", message=str(exc))
            raise TrainingServiceError(exc.code, str(exc), status_code=503) from exc
        except ExportError as exc:
            self._mark_failed_if_active(run_id, stage="export", message=str(exc))
            raise TrainingServiceError(exc.code, str(exc), status_code=503 if exc.code == "EXPORT_RUNTIME_UNAVAILABLE" else 422) from exc
        bundle_dir = output_dir / "bundle"
        try:
            bundle = build_policy_bundle(output_dir=bundle_dir, run_id=run_id, attempt_id=checkpoint.attempt_id, robot_id=adapter.name, observation_dim=self.observation_dim(config, robot), action_dim=robot.dof, export=exported, manifest=run.manifest.model_dump(mode="json"), control_dt=robot.actuation.control_dt, scene_id=config.scene_id, algorithm=config.ppo.algorithm)
            bundle_artifact = self._register_file_artifact(run_id=run_id, attempt_id=checkpoint.attempt_id, kind="policy_bundle", path=bundle_dir.parent / "policy_bundle.tar.gz", content_type="application/gzip")
        except (ExportError, TrainingServiceError):
            self._mark_failed_if_active(run_id, stage="export", message="Policy bundle assembly or publication failed")
            raise
        self.export_results[run_id] = exported
        self.bundle_dirs[run_id] = bundle_dir
        if bundle_artifact is not None:
            self.artifact_ids.setdefault(run_id, []).append(bundle_artifact.artifact_id)
        bundle = bundle.model_copy(update={"artifact_ids": list(self.artifact_ids.get(run_id, []))})
        self.bundles[run_id] = bundle
        self._persist_state(state.model_copy(update={"attempt_id": checkpoint.attempt_id, "training_config": config, "checkpoint": checkpoint, "export_metadata": exported.metadata, "export_files": exported.files, "bundle": bundle, "artifact_ids": list(self.artifact_ids.get(run_id, [])), "updated_at": utc_now()}))
        self.run_service.transition_run(run_id=run_id, target=RunStatus.EXPORTED, stage="export", message="JIT and ONNX export completed")
        return bundle

    def _external_export_result(self, execution, *, output_dir: Path, input_dim: int, output_dim: int, action_scale: float, provider_name: str = "external") -> ExportResult:
        allowed = {".onnx": "onnx", ".pt": "torchscript", ".torchscript": "torchscript"}
        files = [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed and path.name != "export_meta.json"]
        if not files:
            files = [path for path in (getattr(execution, "outputs", {}) or {}).values() if path.is_file() and path.suffix.lower() in allowed]
        if not files:
            raise ExportError("ISAAC_EXPORT_POLICY_MISSING", "Isaac export did not produce a deployable ONNX or TorchScript policy")
        manifest_path = getattr(execution, "manifest_path", None)
        if manifest_path is None or not Path(manifest_path).is_file():
            raise ExportError("ISAAC_EXPORT_MANIFEST_MISSING", "Isaac export did not produce an output manifest")
        self._verify_external_manifest(Path(manifest_path), root=output_dir)
        for path in files:
            self._smoke_validate_policy(path, input_dim=input_dim, output_dim=output_dim)
        records = [file_record(path, root=output_dir, format=allowed[path.suffix.lower()]) for path in files if path.is_file()]
        metadata = ExportMetadata(policy_input_dim=input_dim, policy_output_dim=output_dim, action_scale=action_scale, onnx_opset=None, input_name="observation", output_name="action", exporter=f"{provider_name}-export.v1", runtime="isaac_lab", smoke_passed=True, files=records)
        return ExportResult(metadata=metadata, files=records, output_dir=output_dir)

    @staticmethod
    def _verify_external_manifest(manifest_path: Path, *, root: Path) -> None:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            outputs = payload["outputs"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ExportError("ISAAC_EXPORT_MANIFEST_INVALID", f"unable to read export manifest: {exc}") from exc
        if not isinstance(outputs, list) or not outputs:
            raise ExportError("ISAAC_EXPORT_MANIFEST_INVALID", "export manifest contains no outputs")
        for item in outputs:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ExportError("ISAAC_EXPORT_MANIFEST_INVALID", "export manifest contains an invalid output record")
            path = (root / str(item["path"])).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise ExportError("ISAAC_EXPORT_MANIFEST_INVALID", "export manifest references a path outside the export directory") from exc
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != str(item["sha256"]):
                raise ExportError("ISAAC_EXPORT_MANIFEST_INVALID", f"export manifest checksum mismatch: {item['path']}")

    @staticmethod
    def _smoke_validate_policy(path: Path, *, input_dim: int, output_dim: int) -> None:
        try:
            if path.suffix.lower() == ".onnx":
                import numpy as np
                import onnxruntime as ort

                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                input_meta = session.get_inputs()[0]
                shape = [int(dim) if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
                if len(shape) != 2 or shape[1] != input_dim:
                    raise ExportError("EXPORT_SMOKE_FAILED", f"ONNX input shape does not match observation dimension {input_dim}")
                outputs = session.run(None, {input_meta.name: np.zeros(shape, dtype=np.float32)})
                if not outputs or tuple(outputs[0].shape) != (shape[0], output_dim) or not np.isfinite(outputs[0]).all():
                    raise ExportError("EXPORT_SMOKE_FAILED", "ONNX output shape or finite check failed")
                return
            import torch

            model = torch.jit.load(str(path), map_location="cpu").eval()
            result = model(torch.zeros((1, input_dim), dtype=torch.float32))
            if tuple(result.shape) != (1, output_dim) or not bool(torch.isfinite(result).all()):
                raise ExportError("EXPORT_SMOKE_FAILED", "TorchScript output shape or finite check failed")
        except ExportError:
            raise
        except ImportError as exc:
            raise ExportError("EXPORT_RUNTIME_UNAVAILABLE", f"policy smoke runtime is unavailable: {exc}") from exc
        except Exception as exc:
            raise ExportError("EXPORT_SMOKE_FAILED", f"policy smoke inference failed: {exc}") from exc

    def sim2sim(self, *, run_id: str, seeds: tuple[int, int, int] = (20260101, 20260102, 20260103), adapter=None, thresholds: Sim2SimThresholds | None = None) -> Sim2SimReport:
        run, _ = self.run_service.get_run(run_id=run_id, actor=Actor(user_id=self._creator(run_id)))
        adapter_spec = self._adapter_for_run(run)
        if run.status != RunStatus.EXPORTED:
            raise TrainingServiceError("SIM2SIM_STATUS_INVALID", f"run must be EXPORTED, got {run.status}", status_code=409)
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise TrainingServiceError("SIM2SIM_SEED_INVALID", "exactly three distinct seeds are required")
        self.run_service.transition_run(run_id=run_id, target=RunStatus.SIM2SIM_QUEUED, stage="sim2sim", message="Three-seed sim2sim queued")
        self.run_service.transition_run(run_id=run_id, target=RunStatus.SIM2SIM_RUNNING, stage="sim2sim", message="Three-seed sim2sim started")
        state = self._load_state(run_id)
        config = state.training_config or self.configs.get(run_id)
        checkpoint = state.checkpoint or self.checkpoints.get(run_id)
        if config is None or checkpoint is None:
            raise TrainingServiceError("CHECKPOINT_NOT_FOUND", "durable training state is required before sim2sim", status_code=409)
        if state.export_metadata is not None:
            self.configs[run_id] = config
            self.checkpoints[run_id] = checkpoint
            self.export_results[run_id] = ExportResult(metadata=state.export_metadata, files=state.export_files, output_dir=Path(checkpoint.uri).parent / "export")
            if state.bundle is not None:
                self.bundles[run_id] = state.bundle
                self.bundle_dirs[run_id] = Path(checkpoint.uri).parent / "export" / "bundle"
            self.artifact_ids[run_id] = list(state.artifact_ids)
        evaluator = adapter or self._sim2sim_for_robot(adapter_spec)
        if evaluator is None:
            # A real backend must never silently downgrade to the deterministic
            # fake evaluator when a robot has no registered sim2sim adapter.
            if settings.is_deployed or self.sim2sim_adapter is not None:
                raise TrainingServiceError(
                    "SIM2SIM_ADAPTER_UNSUPPORTED",
                    f"no sim2sim adapter is registered for {adapter_spec.name}",
                    status_code=503,
                )
            evaluator = FakeSim2SimAdapter()
        if hasattr(evaluator, "backend") and evaluator.backend in {"unitree_mujoco", "mujoco"}:
            policy_path = self._policy_path(run_id, checkpoint)
            try:
                with external_run_context(run_id=run_id, runtime_root=settings.runtime_root):
                    executions = [evaluator.evaluate(seed=seed, policy_path=policy_path, run_id=run_id) for seed in seeds]
                self._ensure_not_cancelled(run_id)
            except RunnerError as exc:
                self._mark_failed_if_active(run_id, stage="sim2sim", message=str(exc))
                raise TrainingServiceError(exc.code, str(exc), status_code=503) from exc
            from backend.app.domain.contracts import SeedEvaluation
            evaluations = []
            for item in executions:
                video_artifact_id = None
                for artifact_path in item.artifacts.values():
                    if not artifact_path.is_file():
                        continue
                    kind = "sim2sim_video" if artifact_path.suffix.lower() == ".mp4" else "sim2sim_output"
                    registered = self._register_file_artifact(run_id=run_id, attempt_id=run.current_attempt_id, kind=kind, path=artifact_path, content_type="video/mp4" if kind == "sim2sim_video" else "application/octet-stream")
                    if registered is not None:
                        self.artifact_ids.setdefault(run_id, []).append(registered.artifact_id)
                        if kind == "sim2sim_video" and video_artifact_id is None:
                            video_artifact_id = registered.artifact_id
                evaluations.append(SeedEvaluation(seed=item.seed, status=item.status, exit_code=item.exit_code, duration_seconds=item.duration_seconds, metrics=item.metrics, command=item.command, video_artifact_id=video_artifact_id, failure_code=None if item.status == "PASSED" else "SIM2SIM_RUNTIME_FAILED"))
        else:
            evaluations = [evaluator.evaluate(seed=seed) for seed in seeds]
        report = build_sim2sim_report(run_id=run_id, adapter=evaluator.name, backend=evaluator.backend, evaluations=evaluations, thresholds=thresholds)
        self.reports[run_id] = report
        report_artifact = None
        final_bundle_artifact = None
        bundle_dir = self.bundle_dirs.get(run_id) or (Path(checkpoint.uri).parent / "export" / "bundle")
        exported = self.export_results.get(run_id)
        if bundle_dir is not None and exported is not None:
            # Rebuild the archive so the downloadable package contains the
            # exact report that drove the release decision.
            robot = adapter_spec.get_spec()
            final_bundle = build_policy_bundle(output_dir=bundle_dir, run_id=run_id, attempt_id=run.current_attempt_id, robot_id=adapter_spec.name, observation_dim=self.observation_dim(self.configs[run_id], robot), action_dim=robot.dof, export=exported, manifest=run.manifest.model_dump(mode="json"), sim2sim_report=report.model_dump(mode="json"), control_dt=robot.actuation.control_dt, scene_id=self.configs[run_id].scene_id, algorithm=self.configs[run_id].ppo.algorithm)
            report_artifact = self._register_file_artifact(run_id=run_id, attempt_id=run.current_attempt_id, kind="sim2sim_report", path=bundle_dir / "sim2sim_report.json", content_type="application/json")
            final_bundle_artifact = self._register_file_artifact(run_id=run_id, attempt_id=run.current_attempt_id, kind="policy_bundle_final", path=bundle_dir.parent / "policy_bundle.tar.gz", content_type="application/gzip")
            artifact_ids = list(self.artifact_ids.get(run_id, self.bundles.get(run_id, final_bundle).artifact_ids))
            artifact_ids.extend(artifact.artifact_id for artifact in (report_artifact, final_bundle_artifact) if artifact is not None)
            self.artifact_ids[run_id] = artifact_ids
            self.bundles[run_id] = final_bundle.model_copy(update={"artifact_ids": artifact_ids})
        if run_id in self.bundles:
            self.bundles[run_id] = self.bundles[run_id].model_copy(update={"sim2sim_report": report})
        if report.status == "PASSED":
            if run_id in self.bundles:
                self.bundles[run_id] = self.bundles[run_id].model_copy(update={"status": "READY_TO_DOWNLOAD"})
            self.run_service.transition_run(run_id=run_id, target=RunStatus.SIM2SIM_PASSED, stage="sim2sim", message="Three-seed sim2sim passed")
            self.run_service.transition_run(run_id=run_id, target=RunStatus.READY_TO_DOWNLOAD, stage="release", message="Run is ready to download")
        else:
            self.run_service.transition_run(run_id=run_id, target=RunStatus.FAILED, stage="sim2sim", message="Three-seed sim2sim failed")
        self._persist_state(state.model_copy(update={"attempt_id": run.current_attempt_id, "training_config": config, "checkpoint": checkpoint, "export_metadata": exported.metadata if exported else state.export_metadata, "export_files": exported.files if exported else state.export_files, "bundle": self.bundles.get(run_id, state.bundle), "sim2sim_report": report, "artifact_ids": list(self.artifact_ids.get(run_id, state.artifact_ids)), "updated_at": utc_now()}))
        return report

    def _policy_path(self, run_id: str, checkpoint: CheckpointRecord) -> Path:
        exported = self.export_results.get(run_id)
        if exported is not None:
            for item in exported.files:
                candidate = exported.output_dir / item.path
                if candidate.suffix.lower() in {".pt", ".onnx"} and candidate.is_file():
                    return candidate
        return Path(checkpoint.uri)

    def _mark_failed_if_active(self, run_id: str, *, stage: str, message: str) -> None:
        """Best-effort failure transition that respects concurrent cancel.

        Cancellation is authoritative for the current attempt. An external
        process may therefore return a non-zero code after the API has already
        committed ``CANCELLED``; attempting a second terminal transition would
        mask the original runner error with ``RUN_INVALID_TRANSITION``.
        """

        try:
            with self.run_service.uow:
                run = self.run_service.uow.runs.get(run_id)
            if run is None or run.status == RunStatus.CANCELLED:
                return
            self.run_service.transition_run(run_id=run_id, target=RunStatus.FAILED, stage=stage, message=message)
        except RunServiceError:
            # Another worker may have finalized the run between the read and
            # transition. Preserve the original stage error and durable state.
            return

    def _ensure_not_cancelled(self, run_id: str) -> None:
        with self.run_service.uow:
            run = self.run_service.uow.runs.get(run_id)
        if run is not None and run.status == RunStatus.CANCELLED:
            raise TrainingServiceError("RUN_CANCELLED", "run was cancelled while the external process was running", status_code=409)

    def get_sim2sim_report(self, run_id: str) -> Sim2SimReport | None:
        """Read the report from durable state after an API process restart."""
        report = self.reports.get(run_id)
        if report is not None:
            return report
        state = self._load_state(run_id)
        return state.sim2sim_report

    def _register_file_artifact(self, *, run_id: str, attempt_id: str, kind: str, path: Path, content_type: str) -> ArtifactRecord | None:
        """Publish a worker output through the object-store/application port."""
        if self.artifact_service is None or self.object_store is None:
            return None
        try:
            with self.run_service.uow:
                run = self.run_service.uow.runs.get(run_id)
            if run is None:
                raise TrainingServiceError("RUN_NOT_FOUND", f"run not found: {run_id}", status_code=404)
            object_key = f"projects/{run.project_id}/runs/{run_id}/attempts/{attempt_id}/artifacts/{kind}/{path.name}"
            stored = self.object_store.put_file(object_key, path, content_type=content_type)
            artifact = ArtifactRecord(artifact_id=str(uuid.uuid4()), run_id=run_id, attempt_id=attempt_id, kind=kind, object_key=str(stored["key"]), sha256=str(stored["sha256"]), size_bytes=int(stored["size_bytes"]), content_type=content_type, created_at=utc_now())
            return self.artifact_service.register(artifact)
        except Exception as exc:
            raise TrainingServiceError("ARTIFACT_PUBLISH_FAILED", f"failed to publish {kind}: {exc}", status_code=503) from exc

    @staticmethod
    def observation_dim(config: TrainingConfig, robot: RobotSpec) -> int:
        dof = int(robot.dof)
        base = dof * 2 + 3 + 3 + dof + dof
        return base * config.observation.history_length + 6 + 4

    def _adapter_for_run(self, run):
        robot_id = str((run.manifest.robot or {}).get("robot_id", self.robot_adapter.name))
        try:
            return self.robot_registry.get(robot_id)
        except RobotRegistryError as exc:
            raise TrainingServiceError("ROBOT_NOT_FOUND", str(exc), status_code=404) from exc

    def _resolve_config_for_run(self, run, config: TrainingConfig, *, adapter=None) -> TrainingConfig:
        """Fill task/scene from the selected robot's registered task metadata."""

        target = adapter or self._adapter_for_run(run)
        task_id = config.task_id
        if not task_id:
            candidates = self.task_registry.for_robot(target.name)
            if len(candidates) != 1:
                raise TrainingServiceError(
                    "TASK_REQUIRED",
                    f"task_id is required when {target.name} has {len(candidates)} registered tasks",
                    status_code=422,
                )
            task_id = candidates[0].task_id
        try:
            task = self.task_registry.get(task_id)
        except TaskRegistryError as exc:
            raise TrainingServiceError("TASK_NOT_FOUND", str(exc), status_code=422) from exc
        if task.robot_id != target.name:
            raise TrainingServiceError(
                "TASK_ROBOT_MISMATCH",
                f"task {task.task_id} is registered for robot {task.robot_id}, not {target.name}",
                status_code=422,
            )
        return config.model_copy(update={"task_id": task.task_id, "scene_id": config.scene_id or task.scene_id})

    def _resolve_reward_config(self, run, *, robot_id: str, task_id: str):
        """Load the immutable reward payload selected by the Run Manifest."""

        if self.reward_config_store is None:
            return None
        sha256 = run.manifest.reward_config_sha256
        record = self.reward_config_store.get_by_sha256(sha256)
        if record is None:
            if settings.is_deployed:
                raise TrainingServiceError("REWARD_CONFIG_NOT_FOUND", f"reward config {sha256} is not available to the worker", status_code=409)
            return None
        if record.config_sha256 != sha256:
            raise TrainingServiceError("REWARD_CONFIG_HASH_MISMATCH", "stored reward config hash does not match the Run Manifest", status_code=409)
        validation = self.reward_config_store.validate(record.config, robot_id=robot_id, task_id=task_id)
        if not validation.valid:
            raise TrainingServiceError("REWARD_CONFIG_INVALID", validation.model_dump_json(), status_code=422)
        return record.config

    def _provider_for_task(self, task_id: str, *, robot_id: str | None = None):
        # A selected backend is authoritative for the current worker. This
        # keeps the development ``fake_smoke`` mode from accidentally picking
        # an installed external provider merely because a task is registered.
        try:
            task = self.task_registry.get(task_id)
        except TaskRegistryError:
            return None
        selected = self.training_runner
        if selected is None:
            selected = self.training_provider_registry.get(task.training_provider) or self.training_provider_registry.get(task_id)
        if selected is None:
            return None
        supports = getattr(selected, "supports", None)
        if callable(supports) and robot_id is not None and not supports(robot_id=robot_id, task_id=task_id):
            raise TrainingServiceError(
                "TRAINING_PROVIDER_ROBOT_UNSUPPORTED",
                f"training provider {getattr(selected, 'name', type(selected).__name__)} does not support {robot_id}/{task_id}",
                status_code=422,
            )
        return selected

    def training_provider_for_run(self, *, run_id: str, config: TrainingConfig | None = None, actor: Actor | None = None):
        """Resolve the provider that is allowed to execute a Run.

        Provider selection is scoped by both the Run's RobotSpec and its
        registered task.  API and worker guards use this method before doing
        any external process work, so a globally configured G1 provider cannot
        accidentally be treated as a valid backend for another robot.
        """

        effective_actor = actor or Actor(user_id=self._creator(run_id))
        run, _attempts = self.run_service.get_run(run_id=run_id, actor=effective_actor)
        adapter = self._adapter_for_run(run)
        candidate = config
        if candidate is None:
            state = self._load_state(run_id)
            candidate = state.training_config or self.configs.get(run_id)
        if candidate is None:
            return None
        effective_config = self._resolve_config_for_run(run, candidate, adapter=adapter)
        return self._provider_for_task(effective_config.task_id, robot_id=adapter.name)

    def _sim2sim_for_robot(self, adapter):
        spec = adapter.get_spec()
        selected = self.sim2sim_registry.get(spec.sim2sim_adapter)
        if selected is not None:
            return selected
        # A compatibility adapter may be supplied directly by older callers,
        # but it is only valid when its declared name matches the selected
        # RobotSpec. Never run a G1 evaluator for another robot by fallback.
        if self.sim2sim_adapter is not None and getattr(self.sim2sim_adapter, "name", None) == spec.sim2sim_adapter:
            return self.sim2sim_adapter
        return None

    def has_sim2sim_adapter_for_robot(self, robot_id: str) -> bool:
        """Return whether the selected robot has a concrete evaluator.

        API guards use this method before entering the synchronous path. The
        lookup is intentionally robot-scoped so a configured G1 evaluator
        cannot make an H1 run appear executable.
        """

        try:
            adapter = self.robot_registry.get(robot_id)
        except RobotRegistryError:
            return False
        return self._sim2sim_for_robot(adapter) is not None

    def _creator(self, run_id: str) -> str:
        # Repositories may allocate a transaction-scoped connection (the
        # PostgreSQL UoW does), so never reach through ``uow.runs`` outside a
        # UnitOfWork context.
        with self.run_service.uow:
            run = self.run_service.uow.runs.get(run_id)
            if run is None:
                raise TrainingServiceError("RUN_NOT_FOUND", f"run not found: {run_id}", status_code=404)
            return run.created_by

    def _load_state(self, run_id: str) -> P3RunState:
        with self.run_service.uow:
            state = self.run_service.uow.p3_states.get(run_id)
        if state is not None:
            if state.training_config is not None:
                self.configs[run_id] = state.training_config
            if state.checkpoint is not None:
                self.checkpoints[run_id] = state.checkpoint
            if state.bundle is not None:
                self.bundles[run_id] = state.bundle
            if state.sim2sim_report is not None:
                self.reports[run_id] = state.sim2sim_report
            self.artifact_ids[run_id] = list(state.artifact_ids)
            return state
        with self.run_service.uow:
            run = self.run_service.uow.runs.get(run_id)
        if run is None:
            raise TrainingServiceError("RUN_NOT_FOUND", f"run not found: {run_id}", status_code=404)
        return P3RunState(run_id=run_id, attempt_id=run.current_attempt_id, updated_at=utc_now())

    def _persist_state(self, state: P3RunState) -> P3RunState:
        with self.run_service.uow:
            return self.run_service.uow.p3_states.upsert(state)


__all__ = ["TrainingResult", "TrainingService", "TrainingServiceError"]
