"""P3 training/play/export orchestration with a deterministic CPU smoke backend."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.policy_exporter import ExportError, ExportResult, TorchPolicyExporter, build_policy_bundle
from backend.app.application.run_service import RunService, utc_now
from backend.app.application.sim2sim_service import FakeSim2SimAdapter, build_sim2sim_report
from backend.app.application.training_validator import validate_training_config
from backend.app.domain.contracts import Actor, ArtifactRecord, CheckpointRecord, MetricPoint, P3RunState, PolicyBundle, Sim2SimReport, Sim2SimThresholds, TrainingConfig
from backend.app.domain.state_machine import RunStatus


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
    def __init__(self, *, run_service: RunService, robot_adapter: UnitreeG1Adapter, workspace: Path, artifact_service=None, object_store=None) -> None:
        self.run_service = run_service
        self.robot_adapter = robot_adapter
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifact_service = artifact_service
        self.object_store = object_store
        self.configs: dict[str, TrainingConfig] = {}
        self.checkpoints: dict[str, CheckpointRecord] = {}
        self.bundles: dict[str, PolicyBundle] = {}
        self.reports: dict[str, Sim2SimReport] = {}
        self.export_results: dict[str, ExportResult] = {}
        self.bundle_dirs: dict[str, Path] = {}
        self.artifact_ids: dict[str, list[str]] = {}

    def prepare_training(self, *, run_id: str, config: TrainingConfig, actor: Actor | None = None):
        """Validate and reserve a run for a training worker.

        This method intentionally stops at ``TRAINING_PREPARING`` so an
        asynchronous dispatcher can enqueue the job before GPU work begins.
        The synchronous smoke path advances the final step itself.
        """
        effective_actor = actor or Actor(user_id=self._creator(run_id))
        run, _attempts = self.run_service.get_run(run_id=run_id, actor=effective_actor)
        validation = validate_training_config(config, self.robot_adapter.get_spec())
        if not validation.valid:
            raise TrainingServiceError("TRAIN_CONFIG_INVALID", validation.model_dump_json())
        self.configs[run_id] = config
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
        if run.status == RunStatus.TRAINING_PREPARING:
            self.run_service.transition_run(run_id=run_id, target=RunStatus.TRAINING, stage="training", message="P3 smoke training started")
        output_dir = self.workspace / "runs" / run_id / attempt.attempt_id
        output_dir.mkdir(parents=True, exist_ok=True)
        obs_dim = self.observation_dim(config)
        action_dim = self.robot_adapter.get_spec().dof
        checkpoint_payload = {"format": "g1_mimic_checkpoint.v1", "run_id": run_id, "attempt_id": attempt.attempt_id, "task_id": config.task_id, "observation_dim": obs_dim, "action_dim": action_dim, "iteration": config.ppo.max_iterations, "worker_id": worker_id}
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
        self.run_service.transition_run(run_id=run_id, target=RunStatus.TRAINING_SUCCEEDED, stage="training", message="P3 smoke training completed")
        return TrainingResult(checkpoint=checkpoint, metrics=metrics, output_dir=output_dir, artifacts=artifacts)

    def export(self, *, run_id: str, exporter=None) -> PolicyBundle:
        run, _attempts = self.run_service.get_run(run_id=run_id, actor=Actor(user_id=self._creator(run_id)))
        if run.status != RunStatus.TRAINING_SUCCEEDED:
            raise TrainingServiceError("EXPORT_STATUS_INVALID", f"run must be TRAINING_SUCCEEDED, got {run.status}", status_code=409)
        state = self._load_state(run_id)
        config = state.training_config or self.configs.get(run_id)
        checkpoint = state.checkpoint or self.checkpoints.get(run_id)
        if config is None or checkpoint is None:
            raise TrainingServiceError("CHECKPOINT_NOT_FOUND", "training checkpoint and config are required before export", status_code=409)
        self.run_service.transition_run(run_id=run_id, target=RunStatus.EXPORTING, stage="export", message="Export started")
        output_dir = Path(checkpoint.uri).parent / "export"
        export_impl = exporter or TorchPolicyExporter()
        try:
            exported = export_impl.export(output_dir=output_dir, input_dim=self.observation_dim(config), output_dim=self.robot_adapter.get_spec().dof, action_scale=self.robot_adapter.get_spec().actuation.action_scale)
        except ExportError as exc:
            self.run_service.transition_run(run_id=run_id, target=RunStatus.FAILED, stage="export", message=str(exc))
            raise TrainingServiceError(exc.code, str(exc), status_code=503 if exc.code == "EXPORT_RUNTIME_UNAVAILABLE" else 422) from exc
        bundle_dir = output_dir / "bundle"
        try:
            bundle = build_policy_bundle(output_dir=bundle_dir, run_id=run_id, attempt_id=checkpoint.attempt_id, robot_id=self.robot_adapter.name, observation_dim=self.observation_dim(config), action_dim=self.robot_adapter.get_spec().dof, export=exported, manifest=run.manifest.model_dump(mode="json"))
            bundle_artifact = self._register_file_artifact(run_id=run_id, attempt_id=checkpoint.attempt_id, kind="policy_bundle", path=bundle_dir.parent / "policy_bundle.tar.gz", content_type="application/gzip")
        except (ExportError, TrainingServiceError):
            self.run_service.transition_run(run_id=run_id, target=RunStatus.FAILED, stage="export", message="Policy bundle assembly or publication failed")
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

    def sim2sim(self, *, run_id: str, seeds: tuple[int, int, int] = (20260101, 20260102, 20260103), adapter=None, thresholds: Sim2SimThresholds | None = None) -> Sim2SimReport:
        run, _ = self.run_service.get_run(run_id=run_id, actor=Actor(user_id=self._creator(run_id)))
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
        evaluator = adapter or FakeSim2SimAdapter()
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
            final_bundle = build_policy_bundle(output_dir=bundle_dir, run_id=run_id, attempt_id=run.current_attempt_id, robot_id=self.robot_adapter.name, observation_dim=self.observation_dim(self.configs[run_id]), action_dim=self.robot_adapter.get_spec().dof, export=exported, manifest=run.manifest.model_dump(mode="json"), sim2sim_report=report.model_dump(mode="json"))
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
    def observation_dim(config: TrainingConfig) -> int:
        # G1-specific dimensions are checked by the adapter; this is the
        # deterministic schema calculation shared by train/export metadata.
        base = 29 * 2 + 3 + 3 + 29 + 29
        return base * config.observation.history_length + 6 + 4

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
