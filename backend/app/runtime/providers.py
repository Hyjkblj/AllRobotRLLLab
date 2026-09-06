"""Training provider boundary.

Unitree RL Lab is intentionally an optional provider. The platform can select
another provider for a task without changing application services.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from backend.app.runtime.contracts import ExternalRunResult, RunnerError, TrainingExecution
from backend.app.runtime.isaac_runner import IsaacLabRunner
from backend.app.runtime.process import command_from_env, run_external, write_output_manifest
from backend.app.runtime.registry import RuntimeRegistry
from backend.app.domain.contracts import RunManifest


class TrainingProvider(Protocol):
    name: str

    def train(self, *, run_id: str, task_id: str, motion_path: Path, config: dict[str, Any], output_dir: Path | None = None) -> TrainingExecution: ...

    def export(self, *, checkpoint_path: Path, task_id: str, output_dir: Path | None = None) -> ExternalRunResult: ...

    def play(self, *, checkpoint_path: Path, task_id: str, output_dir: Path | None = None) -> ExternalRunResult: ...


class UnitreeRLLabProvider:
    """G1 provider backed by the externally installed Unitree RL Lab."""

    name = "unitree_rl_lab"
    supported_robot_ids = frozenset({"unitree_g1_29dof"})
    # Platform IDs are stable API identifiers. Unitree RL Lab registers
    # Gymnasium IDs, which are an upstream implementation detail.
    task_id_map = {
        "g1_mimic": "Unitree-G1-29dof-Mimic-Gangnanm-Style",
    }

    def __init__(self, runner: IsaacLabRunner) -> None:
        self.runner = runner

    def supports(self, *, robot_id: str, task_id: str) -> bool:
        """Keep the vendor-specific provider from being used for another robot."""

        return robot_id in self.supported_robot_ids and task_id == "g1_mimic"

    @classmethod
    def upstream_task_id(cls, task_id: str) -> str:
        try:
            return cls.task_id_map[task_id]
        except KeyError as exc:
            raise RunnerError("UNITREE_TASK_UNSUPPORTED", f"Unitree RL Lab has no mapping for platform task {task_id}") from exc

    def train(self, **kwargs: Any) -> TrainingExecution:
        return self.runner.train(**{**kwargs, "task_id": self.upstream_task_id(str(kwargs["task_id"]))})

    def export(self, **kwargs: Any) -> ExternalRunResult:
        return self.runner.export(**{**kwargs, "task_id": self.upstream_task_id(str(kwargs["task_id"]))})

    def play(self, **kwargs: Any) -> ExternalRunResult:
        return self.runner.play(**{**kwargs, "task_id": self.upstream_task_id(str(kwargs["task_id"]))})


class NativeIsaacLabProvider:
    """Platform-owned Isaac Lab entrypoint contract.

    A deployment must provide explicit command templates. The provider never
    falls back to Unitree RL Lab, which makes the boundary auditable while the
    native task implementation is developed in the selected Isaac runtime.
    """

    name = "native_isaac_lab"
    version = "native-isaac-provider.v1"

    def supports(self, *, robot_id: str, task_id: str) -> bool:
        """Native tasks are resolved by the selected Isaac task registry."""

        return bool(robot_id.strip() and task_id.strip())

    def __init__(self, *, registry: RuntimeRegistry, workspace: Path, timeout_seconds: float = 24 * 3600) -> None:
        self.registry = registry
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def _checks(self):
        return self.registry.require("isaac_lab"), self.registry.require("isaac_sim")

    @staticmethod
    def _require_command(env_name: str, *, default: list[str] | None = None) -> tuple[str, ...]:
        command = command_from_env(env_name, default=default)
        if not command:
            raise RunnerError("NATIVE_ISAAC_COMMAND_MISSING", f"{env_name} must be configured for the native Isaac provider")
        return command

    def train(self, *, run_id: str, task_id: str, motion_path: Path, config: dict[str, Any], output_dir: Path | None = None) -> TrainingExecution:
        isaac_lab, isaac_sim = self._checks()
        target = Path(output_dir or self.workspace / run_id / "train").resolve()
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "training_config.json"
        command_template = self._require_command("NATIVE_ISAAC_TRAIN_COMMAND")
        # ``_run_manifest`` is injected by TrainingService. Keep it out of the
        # training config payload because the Isaac task entrypoint consumes
        # the manifest as a separate, immutable contract file.
        raw_manifest = config.get("_run_manifest")
        if not isinstance(raw_manifest, dict):
            raise RunnerError(
                "NATIVE_ISAAC_MANIFEST_MISSING",
                "native Isaac training requires the frozen Run Manifest",
            )
        try:
            manifest = RunManifest.model_validate(raw_manifest)
        except Exception as exc:
            raise RunnerError(
                "NATIVE_ISAAC_MANIFEST_INVALID",
                f"native Isaac training received an invalid Run Manifest: {exc}",
            ) from exc
        if manifest.run_id != run_id:
            raise RunnerError(
                "NATIVE_ISAAC_MANIFEST_MISMATCH",
                f"Run Manifest run_id {manifest.run_id} does not match requested run {run_id}",
            )
        config_payload = {key: value for key, value in config.items() if key != "_run_manifest"}
        config_path.write_text(json.dumps({"run_id": run_id, "task_id": task_id, "motion_path": str(Path(motion_path).resolve()), "config": config_payload}, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        manifest_path = target / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        command = tuple(item.format(run_id=run_id, task=task_id, motion=str(motion_path), output=str(target), config=str(config_path), manifest=str(manifest_path)) for item in command_template)
        result = run_external(stage="native_isaac_train", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_RUN_ID": run_id, "ALLROBOTRL_MOTION": str(motion_path), "ALLROBOTRL_OUTPUT": str(target), "ALLROBOTRL_CONFIG": str(config_path), "ALLROBOTRL_MANIFEST": str(manifest_path)})
        checkpoint = IsaacLabRunner._find_checkpoint(target)
        if checkpoint is None:
            raise RunnerError("ISAAC_CHECKPOINT_MISSING", "native Isaac provider completed without a checkpoint", details={"workspace": str(target)})
        metrics = IsaacLabRunner._read_metrics(target)
        manifest = write_output_manifest(target, stage="native_isaac_train", outputs=[checkpoint, *IsaacLabRunner._metric_files(target)], metadata={"runtime": isaac_lab.as_dict(), "runtime_dependencies": [isaac_sim.as_dict()], "provider": self.name, "adapter_version": self.version, "task_id": task_id})
        return TrainingExecution(checkpoint_path=checkpoint, iteration=int(metrics[-1].get("iteration", config.get("ppo", {}).get("max_iterations", 0))) if metrics else int(config.get("ppo", {}).get("max_iterations", 0)), metrics=metrics, result=ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {"checkpoint": checkpoint}, manifest))

    def export(self, *, checkpoint_path: Path, task_id: str, output_dir: Path | None = None) -> ExternalRunResult:
        isaac_lab, isaac_sim = self._checks()
        target = Path(output_dir or checkpoint_path.parent / "export").resolve()
        target.mkdir(parents=True, exist_ok=True)
        command = tuple(item.format(checkpoint=str(checkpoint_path), task=task_id, output=str(target)) for item in self._require_command("NATIVE_ISAAC_EXPORT_COMMAND"))
        result = run_external(stage="native_isaac_export", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_CHECKPOINT": str(checkpoint_path), "ALLROBOTRL_OUTPUT": str(target)})
        outputs = [path for path in target.rglob("*") if path.is_file() and path.name != "isaac_export.json"]
        if not outputs:
            raise RunnerError("ISAAC_EXPORT_OUTPUT_MISSING", "native Isaac provider produced no export files")
        manifest = write_output_manifest(target, stage="native_isaac_export", outputs=outputs, metadata={"runtime": isaac_lab.as_dict(), "runtime_dependencies": [isaac_sim.as_dict()], "provider": self.name, "adapter_version": self.version, "task_id": task_id})
        return ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {path.name: path for path in outputs}, manifest)

    def play(self, *, checkpoint_path: Path, task_id: str, output_dir: Path | None = None) -> ExternalRunResult:
        isaac_lab, isaac_sim = self._checks()
        target = Path(output_dir or checkpoint_path.parent / "play").resolve()
        target.mkdir(parents=True, exist_ok=True)
        command = tuple(item.format(checkpoint=str(checkpoint_path), task=task_id, output=str(target)) for item in self._require_command("NATIVE_ISAAC_PLAY_COMMAND"))
        result = run_external(stage="native_isaac_play", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_CHECKPOINT": str(checkpoint_path), "ALLROBOTRL_OUTPUT": str(target)})
        outputs = [path for path in target.rglob("*") if path.is_file()]
        manifest = write_output_manifest(target, stage="native_isaac_play", outputs=outputs, metadata={"runtime": isaac_lab.as_dict(), "runtime_dependencies": [isaac_sim.as_dict()], "provider": self.name, "adapter_version": self.version, "task_id": task_id})
        return ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {path.name: path for path in outputs}, manifest)


__all__ = ["NativeIsaacLabProvider", "TrainingProvider", "UnitreeRLLabProvider"]
