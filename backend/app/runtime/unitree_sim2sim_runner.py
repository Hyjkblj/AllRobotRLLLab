"""Unitree MuJoCo three-seed sim2sim adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.app.runtime.contracts import RunnerError, Sim2SimExecution
from backend.app.runtime.process import command_from_env, run_external, write_output_manifest
from backend.app.runtime.registry import RuntimeRegistry


class UnitreeMuJoCoRunner:
    name = "unitree_g1_mujoco"
    backend = "unitree_mujoco"
    version = "unitree-mujoco-runner.v1"

    def __init__(self, *, registry: RuntimeRegistry, workspace: Path, timeout_seconds: float = 900) -> None:
        self.registry = registry
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def evaluate(self, *, seed: int, policy_path: Path | None = None, run_id: str = "run", output_dir: Path | None = None) -> Sim2SimExecution:
        check = self.registry.require("unitree_mujoco")
        target = Path(output_dir or self.workspace / run_id / f"seed-{seed}").resolve()
        target.mkdir(parents=True, exist_ok=True)
        if policy_path is None:
            raise RunnerError("SIM2SIM_POLICY_MISSING", "a policy path is required for Unitree MuJoCo evaluation")
        default = [str(Path(check.path or ".") / "simulate" / "build" / "unitree_mujoco"), "--seed", str(seed), "--policy", str(policy_path)]
        command = command_from_env("UNITREE_SIM2SIM_COMMAND", default=default)
        command = tuple(item.format(seed=seed, policy=str(policy_path), output=str(target), run_id=run_id) for item in (command or default))
        started = time.perf_counter()
        try:
            result = run_external(stage=f"sim2sim_seed_{seed}", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_RUN_ID": run_id, "ALLROBOTRL_SEED": str(seed), "ALLROBOTRL_POLICY": str(policy_path), "ALLROBOTRL_OUTPUT": str(target)})
        except RunnerError as exc:
            return Sim2SimExecution(seed, "FAILED", int(exc.details.get("return_code", 1)), time.perf_counter() - started, {}, list(command), stderr=str(exc))
        metrics = self._read_metrics(target)
        required = {"survival_rate", "joint_rmse_rad", "root_position_rmse_m", "orientation_error_deg", "saturation_ratio", "foot_slip_mps"}
        if not required.issubset(metrics):
            return Sim2SimExecution(seed, "FAILED", 0, time.perf_counter() - started, metrics, list(command), stderr="sim2sim metrics.json is missing required keys")
        artifacts = {path.name: path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in {".mp4", ".csv", ".json", ".png"}}
        try:
            write_output_manifest(target, stage=f"sim2sim_seed_{seed}", outputs=artifacts.values(), metadata={"runtime": check.as_dict(), "adapter_version": self.version, "seed": seed})
        except RunnerError:
            artifacts = {}
        return Sim2SimExecution(seed, "PASSED", 0, time.perf_counter() - started, metrics, list(command), artifacts)

    @staticmethod
    def _read_metrics(root: Path) -> dict[str, float]:
        for path in (root / "metrics.json", *sorted(root.rglob("metrics.json"))):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict):
                result: dict[str, float] = {}
                for key, value in payload.items():
                    try:
                        result[str(key)] = float(value)
                    except (TypeError, ValueError):
                        continue
                return result
        return {}


__all__ = ["UnitreeMuJoCoRunner"]
