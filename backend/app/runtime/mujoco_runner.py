"""Generic command-backed MuJoCo sim2sim adapter.

Unlike the Unitree adapter this runner has no assumptions about DDS, sensor
ordering, or controller layout. A robot package supplies the command through
its deployment environment and reports the standard metrics contract.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from backend.app.runtime.contracts import RunnerError, Sim2SimExecution
from backend.app.runtime.process import command_from_env, run_external, write_output_manifest


class MuJoCoRunner:
    backend = "mujoco"
    version = "mujoco-runner.v1"

    def __init__(self, *, name: str, command_env: str = "MUJOCO_SIM2SIM_COMMAND", workspace: Path, timeout_seconds: float = 900) -> None:
        self.name = name
        self.command_env = command_env
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def evaluate(self, *, seed: int, policy_path: Path | None = None, run_id: str = "run", output_dir: Path | None = None) -> Sim2SimExecution:
        target = Path(output_dir or self.workspace / run_id / f"seed-{seed}").resolve()
        target.mkdir(parents=True, exist_ok=True)
        if policy_path is None:
            raise RunnerError("SIM2SIM_POLICY_MISSING", "a policy path is required for MuJoCo evaluation")
        default = ["python", "-m", "mujoco_sim2sim", "--seed", str(seed), "--policy", str(policy_path)]
        command = command_from_env(self.command_env, default=default) or tuple(default)
        command = tuple(item.format(seed=seed, policy=str(policy_path), output=str(target), run_id=run_id) for item in command)
        started = time.perf_counter()
        try:
            result = run_external(stage=f"{self.name}_seed_{seed}", workspace=target, command=command, timeout_seconds=self.timeout_seconds, env={"ALLROBOTRL_RUN_ID": run_id, "ALLROBOTRL_SEED": str(seed), "ALLROBOTRL_POLICY": str(policy_path), "ALLROBOTRL_OUTPUT": str(target)})
        except RunnerError as exc:
            return Sim2SimExecution(seed, "FAILED", int(exc.details.get("return_code", 1)), time.perf_counter() - started, {}, list(command), stderr=str(exc))
        metrics = self._read_metrics(target)
        required = {"survival_rate", "joint_rmse_rad", "root_position_rmse_m", "orientation_error_deg", "saturation_ratio", "foot_slip_mps"}
        if not required.issubset(metrics) or any(not math.isfinite(metrics[name]) for name in required):
            return Sim2SimExecution(seed, "FAILED", result.return_code, time.perf_counter() - started, metrics, list(command), stderr="sim2sim metrics.json is missing required keys")
        artifacts = {path.name: path for path in target.rglob("*") if path.is_file() and "manifest" not in path.parts}
        manifest = write_output_manifest(target, stage=f"{self.name}_seed_{seed}", outputs=artifacts.values(), metadata={"adapter_version": self.version, "seed": seed})
        artifacts[manifest.name] = manifest
        return Sim2SimExecution(seed, "PASSED", result.return_code, time.perf_counter() - started, metrics, list(command), artifacts)

    @staticmethod
    def _read_metrics(root: Path) -> dict[str, float]:
        path = root / "metrics.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        result: dict[str, float] = {}
        for key, value in payload.items():
            try:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        result[str(key)] = numeric
            except (TypeError, ValueError):
                continue
        return result


__all__ = ["MuJoCoRunner"]
