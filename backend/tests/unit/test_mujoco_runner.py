from pathlib import Path
import pytest

from backend.app.runtime.mujoco_runner import MuJoCoRunner
from backend.app.runtime.contracts import RunnerError, RuntimeCheck
from backend.app.runtime.unitree_sim2sim_runner import UnitreeMuJoCoRunner
from backend.app.runtime.registry import RuntimeRegistry


def test_generic_mujoco_runner_uses_robot_specific_command(tmp_path: Path, monkeypatch) -> None:
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"policy")
    script = tmp_path / "simulate.py"
    script.write_text(
        "import json; json.dump({'survival_rate': .99, 'joint_rmse_rad': .1, 'root_position_rmse_m': .05, 'orientation_error_deg': 3, 'saturation_ratio': .01, 'foot_slip_mps': .02}, open('metrics.json', 'w'))",
        encoding="utf-8",
    )
    command = f'python "{script}"'
    monkeypatch.setenv("H1_SIM2SIM_COMMAND", command)
    runner = MuJoCoRunner(name="mujoco_h1", command_env="H1_SIM2SIM_COMMAND", workspace=tmp_path / "runs")
    result = runner.evaluate(seed=11, policy_path=policy, run_id="run-1")
    assert result.status == "PASSED"
    assert result.metrics["survival_rate"] == 0.99
    assert "metrics.json" in result.artifacts
    assert (tmp_path / "runs" / "run-1" / "seed-11" / "manifest" / "mujoco_h1_seed_11.json").is_file()


def test_unitree_runner_uses_project_evaluator_by_default(tmp_path: Path, monkeypatch) -> None:
    registry = RuntimeRegistry(registration_path=tmp_path / "registrations.json")
    runtime = tmp_path / "unitree"
    runtime.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    registry.register("unitree_mujoco", path=runtime, python=str(python), revision="ae6a840")
    monkeypatch.setattr(registry, "require", lambda _name: RuntimeCheck("unitree_mujoco", True, True, path=str(runtime), python=str(python)))
    monkeypatch.delenv("UNITREE_SIM2SIM_COMMAND", raising=False)
    runner = UnitreeMuJoCoRunner(registry=registry, workspace=tmp_path / "runs")
    with pytest.raises(RunnerError) as raised:
        runner.evaluate(seed=1, policy_path=tmp_path / "policy.onnx")
    assert raised.value.code == "SIM2SIM_MOTION_MISSING"


def test_unitree_runner_default_command_includes_policy_motion_and_runtime(tmp_path: Path, monkeypatch) -> None:
    registry = RuntimeRegistry(registration_path=tmp_path / "registrations.json")
    runtime = tmp_path / "unitree"
    runtime.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    registry.register("unitree_mujoco", path=runtime, python=str(python), revision="ae6a840")
    monkeypatch.setattr(registry, "require", lambda _name: RuntimeCheck("unitree_mujoco", True, True, path=str(runtime), python=str(python)))
    monkeypatch.delenv("UNITREE_SIM2SIM_COMMAND", raising=False)
    captured = {}
    def fake_run_external(**kwargs):
        captured.update(kwargs)
        Path(kwargs["workspace"], "metrics.json").write_text('{"survival_rate":1,"joint_rmse_rad":0.1,"root_position_rmse_m":0.1,"orientation_error_deg":1,"saturation_ratio":0,"foot_slip_mps":0}', encoding="utf-8")
        return type("Result", (), {"return_code": 0})()
    monkeypatch.setattr("backend.app.runtime.unitree_sim2sim_runner.run_external", fake_run_external)
    policy = tmp_path / "policy.onnx"
    motion = tmp_path / "motion.npz"
    policy.write_bytes(b"policy")
    motion.write_bytes(b"motion")
    result = UnitreeMuJoCoRunner(registry=registry, workspace=tmp_path / "runs").evaluate(seed=7, policy_path=policy, motion_path=motion)
    assert result.status == "PASSED"
    assert captured["command"][:3] == (str(python), "-m", "apps.mujoco_sim2sim.g1")
    assert captured["env"]["ALLROBOTRL_MOTION"] == str(motion)
