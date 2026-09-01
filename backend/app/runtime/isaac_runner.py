"""Isaac Lab/Unitree RL Lab process adapter."""

from __future__ import annotations

import json
import re
import shutil
import os
from pathlib import Path
from typing import Any

from backend.app.runtime.contracts import ExternalRunResult, RunnerError, TrainingExecution
from backend.app.runtime.process import command_from_env, run_external, write_output_manifest
from backend.app.runtime.registry import RuntimeRegistry


class IsaacLabRunner:
    name = "isaac_lab"
    version = "isaac-lab-runner.v1"

    def __init__(self, *, registry: RuntimeRegistry, workspace: Path, timeout_seconds: float = 24 * 3600) -> None:
        self.registry = registry
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def train(self, *, run_id: str, task_id: str, motion_path: Path, config: dict[str, Any], output_dir: Path | None = None) -> TrainingExecution:
        check = self.registry.require("isaac_lab")
        target = Path(output_dir or self.workspace / run_id / "train").resolve()
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "training_config.json"
        config_path.write_text(json.dumps({"run_id": run_id, "task_id": task_id, "motion_path": str(Path(motion_path).resolve()), "config": config}, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        rl_path = Path(os.getenv("UNITREE_RL_LAB_PATH", "")).expanduser() if os.getenv("UNITREE_RL_LAB_PATH", "").strip() else Path(check.path or ".")
        default = [check.python or "python", str(rl_path / "scripts" / "rsl_rl" / "train.py"), "--headless", "--task", task_id]
        command = command_from_env("ISAAC_TRAIN_COMMAND", default=default)
        command = tuple(item.format(run_id=run_id, task=task_id, motion=str(motion_path), output=str(target), config=str(config_path)) for item in (command or default))
        result = run_external(stage="isaac_train", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_RUN_ID": run_id, "ALLROBOTRL_MOTION": str(motion_path), "ALLROBOTRL_OUTPUT": str(target), "ALLROBOTRL_CONFIG": str(config_path)})
        checkpoint = self._find_checkpoint(target)
        if checkpoint is None:
            raise RunnerError("ISAAC_CHECKPOINT_MISSING", "Isaac Lab completed without a checkpoint", details={"workspace": str(target)})
        metrics = self._read_metrics(target)
        manifest = write_output_manifest(target, stage="isaac_train", outputs=[checkpoint, *self._metric_files(target)], metadata={"runtime": check.as_dict(), "adapter_version": self.version, "task_id": task_id})
        result = ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {"checkpoint": checkpoint}, manifest)
        iteration = int(metrics[-1].get("iteration", config.get("ppo", {}).get("max_iterations", 0))) if metrics else int(config.get("ppo", {}).get("max_iterations", 0))
        return TrainingExecution(checkpoint_path=checkpoint, iteration=iteration, metrics=metrics, result=result)

    def export(self, *, checkpoint_path: Path, task_id: str, output_dir: Path | None = None) -> ExternalRunResult:
        check = self.registry.require("isaac_lab")
        target = Path(output_dir or checkpoint_path.parent / "export").resolve()
        target.mkdir(parents=True, exist_ok=True)
        rl_path = Path(os.getenv("UNITREE_RL_LAB_PATH", "")).expanduser() if os.getenv("UNITREE_RL_LAB_PATH", "").strip() else Path(check.path or ".")
        default = [check.python or "python", str(rl_path / "scripts" / "rsl_rl" / "play.py"), "--task", task_id, "--checkpoint", str(checkpoint_path)]
        command = command_from_env("ISAAC_EXPORT_COMMAND", default=default)
        command = tuple(item.format(checkpoint=str(checkpoint_path), task=task_id, output=str(target)) for item in (command or default))
        result = run_external(stage="isaac_export", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_CHECKPOINT": str(checkpoint_path), "ALLROBOTRL_OUTPUT": str(target)})
        outputs = [path for path in target.rglob("*") if path.is_file() and path.name != "isaac_export.json"]
        # Unitree RL Lab's play.py writes exported/policy.{pt,onnx} beside
        # the checkpoint. Copy those files into the platform-owned output
        # directory so artifact publication is independent of that layout.
        if not outputs:
            upstream_export = checkpoint_path.parent / "exported"
            if upstream_export.is_dir():
                for source in upstream_export.iterdir():
                    if source.is_file():
                        target_file = target / source.name
                        shutil.copy2(source, target_file)
                outputs = [path for path in target.iterdir() if path.is_file()]
        if not outputs:
            raise RunnerError("ISAAC_EXPORT_OUTPUT_MISSING", "Isaac Lab export produced no files")
        write_output_manifest(target, stage="isaac_export", outputs=outputs, metadata={"runtime": check.as_dict(), "adapter_version": self.version})
        return ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {path.name: path for path in outputs})

    @staticmethod
    def _find_checkpoint(root: Path) -> Path | None:
        candidates = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".json"} and any(token in path.name.lower() for token in ("checkpoint", "model", "policy", "agent")))
        return candidates[0] if candidates else None

    @staticmethod
    def _metric_files(root: Path) -> list[Path]:
        return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".csv", ".jsonl"} and "config" not in path.name.lower()]

    @staticmethod
    def _read_metrics(root: Path) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for path in sorted(root.rglob("metrics*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, list):
                values.extend(item for item in payload if isinstance(item, dict))
            elif isinstance(payload, dict):
                values.append(payload)
        for line in "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*.log") if path.is_file()).splitlines():
            match = re.search(r"(?:iteration|step)\s*[=:]\s*(\d+).*?(?:return|reward)\s*[=:]\s*([-+]?\d+(?:\.\d+)?)", line, re.I)
            if match:
                values.append({"iteration": int(match.group(1)), "name": "train/return", "value": float(match.group(2))})
        return values


__all__ = ["IsaacLabRunner"]
