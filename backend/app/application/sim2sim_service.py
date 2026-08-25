"""Three-seed sim2sim evaluation and threshold aggregation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from backend.app.domain.contracts import SeedEvaluation, Sim2SimReport, Sim2SimThresholds


@dataclass(frozen=True)
class FakeSim2SimAdapter:
    name: str = "unitree_g1_mujoco_smoke"
    backend: str = "fake_smoke"

    def evaluate(self, seed: int) -> SeedEvaluation:
        started = time.perf_counter()
        spread = (seed % 7) * 0.001
        metrics = {"survival_rate": 0.98 - spread, "joint_rmse_rad": 0.12 + spread / 2, "root_position_rmse_m": 0.06 + spread / 4, "orientation_error_deg": 6.0 + spread * 10, "saturation_ratio": 0.01 + spread / 10, "foot_slip_mps": 0.04 + spread / 10}
        return SeedEvaluation(seed=seed, status="PASSED", exit_code=0, duration_seconds=time.perf_counter() - started, metrics=metrics, command=["unitree_g1_mujoco", "--seed", str(seed), "--smoke"])


def build_sim2sim_report(*, run_id: str, adapter: str, backend: str, evaluations: list[SeedEvaluation], thresholds: Sim2SimThresholds | None = None) -> Sim2SimReport:
    policy = thresholds or Sim2SimThresholds()
    failures: list[str] = []
    if len(evaluations) != 3:
        failures.append("SIM2SIM_SEED_COUNT_INVALID")
    for evaluation in evaluations:
        metrics = evaluation.metrics
        if evaluation.status != "PASSED" or evaluation.exit_code != 0:
            failures.append(f"SEED_{evaluation.seed}_PROCESS_FAILED")
        if metrics.get("survival_rate", 0.0) < policy.min_survival_rate:
            failures.append(f"SEED_{evaluation.seed}_SURVIVAL_LOW")
        if metrics.get("joint_rmse_rad", float("inf")) > policy.max_joint_rmse_rad:
            failures.append(f"SEED_{evaluation.seed}_JOINT_RMSE_HIGH")
        if metrics.get("root_position_rmse_m", float("inf")) > policy.max_root_position_rmse_m:
            failures.append(f"SEED_{evaluation.seed}_ROOT_POSITION_HIGH")
        if metrics.get("orientation_error_deg", float("inf")) > policy.max_orientation_error_deg:
            failures.append(f"SEED_{evaluation.seed}_ORIENTATION_HIGH")
        if metrics.get("saturation_ratio", float("inf")) > policy.max_saturation_ratio:
            failures.append(f"SEED_{evaluation.seed}_SATURATION_HIGH")
        if metrics.get("foot_slip_mps", float("inf")) > policy.max_foot_slip_mps:
            failures.append(f"SEED_{evaluation.seed}_FOOT_SLIP_HIGH")
    if evaluations:
        for name in ("survival_rate", "joint_rmse_rad", "root_position_rmse_m", "orientation_error_deg", "saturation_ratio", "foot_slip_mps"):
            values = [evaluation.metrics[name] for evaluation in evaluations if name in evaluation.metrics]
            if len(values) == len(evaluations):
                denominator = max(abs(sum(values) / len(values)), 1e-9)
                if (max(values) - min(values)) / denominator > policy.max_seed_metric_spread:
                    failures.append(f"SEED_SPREAD_HIGH_{name}")
    status = "PASSED" if not failures and len(evaluations) == 3 else "FAILED"
    payload = {"run_id": run_id, "adapter": adapter, "backend": backend, "thresholds": policy.model_dump(mode="json"), "evaluations": [evaluation.model_dump(mode="json") for evaluation in evaluations], "status": status, "hard_failures": failures}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return Sim2SimReport(run_id=run_id, adapter=adapter, backend=backend, thresholds=policy, evaluations=evaluations, status=status, hard_failures=failures, report_sha256=digest)


__all__ = ["FakeSim2SimAdapter", "build_sim2sim_report"]
