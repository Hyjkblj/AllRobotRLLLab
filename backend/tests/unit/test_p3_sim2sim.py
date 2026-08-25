from backend.app.application.sim2sim_service import FakeSim2SimAdapter, build_sim2sim_report
from backend.app.domain.contracts import SeedEvaluation, Sim2SimThresholds


def test_fake_sim2sim_aggregates_three_seeds_to_pass() -> None:
    adapter = FakeSim2SimAdapter()
    report = build_sim2sim_report(run_id="run-1", adapter=adapter.name, backend=adapter.backend, evaluations=[adapter.evaluate(seed) for seed in (20260101, 20260102, 20260103)])
    assert report.status == "PASSED"
    assert len(report.evaluations) == 3
    assert len(report.report_sha256) == 64


def test_sim2sim_threshold_failure_is_diagnostic() -> None:
    evaluation = SeedEvaluation(seed=1, status="PASSED", exit_code=0, duration_seconds=1, metrics={"survival_rate": 0.2, "joint_rmse_rad": 0.9, "root_position_rmse_m": 0.8, "orientation_error_deg": 50, "saturation_ratio": 0.2, "foot_slip_mps": 0.5})
    report = build_sim2sim_report(run_id="run-1", adapter="g1", backend="test", evaluations=[evaluation, evaluation.model_copy(update={"seed": 2}), evaluation.model_copy(update={"seed": 3})], thresholds=Sim2SimThresholds())
    assert report.status == "FAILED"
    assert "SEED_1_SURVIVAL_LOW" in report.hard_failures
    assert "SEED_1_JOINT_RMSE_HIGH" in report.hard_failures

