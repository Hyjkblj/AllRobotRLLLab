import json
from pathlib import Path

import pytest

from backend.app.runtime.contracts import ExternalRunResult, RunnerError
from backend.app.runtime.providers import NativeIsaacLabProvider
from backend.app.runtime.registry import RuntimeRegistry
from backend.app.runtime.isaac_runner import IsaacLabRunner
from backend.app.runtime.providers import UnitreeRLLabProvider
from backend.app.domain.contracts import RunManifest


def test_native_isaac_provider_requires_an_explicit_platform_command(tmp_path: Path, monkeypatch) -> None:
    isaac_lab = tmp_path / "isaac_lab"
    isaac_sim = tmp_path / "isaac_sim"
    isaac_lab.mkdir()
    isaac_sim.mkdir()
    registration = tmp_path / "registrations.json"
    registry = RuntimeRegistry(registration_path=registration)
    registry.register("isaac_lab", path=isaac_lab)
    registry.register("isaac_sim", path=isaac_sim)
    monkeypatch.delenv("NATIVE_ISAAC_TRAIN_COMMAND", raising=False)
    provider = NativeIsaacLabProvider(registry=registry, workspace=tmp_path / "work")
    with pytest.raises(RunnerError) as raised:
        provider.train(run_id="run", task_id="task", motion_path=tmp_path / "motion.npz", config={})
    assert raised.value.code == "NATIVE_ISAAC_COMMAND_MISSING"


def test_unitree_provider_is_scoped_to_g1() -> None:
    runner = IsaacLabRunner.__new__(IsaacLabRunner)
    provider = UnitreeRLLabProvider(runner)
    assert provider.supports(robot_id="unitree_g1_29dof", task_id="g1_mimic")
    assert not provider.supports(robot_id="unitree_h1_19dof", task_id="h1_mimic")
    assert provider.upstream_task_id("g1_mimic") == "Unitree-G1-29dof-Mimic-Gangnanm-Style"


def test_unitree_provider_passes_upstream_task_id_to_runner() -> None:
    class Runner:
        def train(self, **kwargs):
            return kwargs

        def export(self, **kwargs):
            return kwargs

        def play(self, **kwargs):
            return kwargs

    provider = UnitreeRLLabProvider(Runner())
    result = provider.train(task_id="g1_mimic", run_id="run", motion_path=Path("motion"), config={})
    assert result["task_id"] == "Unitree-G1-29dof-Mimic-Gangnanm-Style"


def test_native_isaac_provider_writes_and_passes_run_manifest(tmp_path: Path, monkeypatch) -> None:
    isaac_lab = tmp_path / "isaac_lab"
    isaac_sim = tmp_path / "isaac_sim"
    isaac_lab.mkdir()
    isaac_sim.mkdir()
    registry = RuntimeRegistry(registration_path=tmp_path / "registrations.json")
    registry.register("isaac_lab", path=isaac_lab)
    registry.register("isaac_sim", path=isaac_sim)
    monkeypatch.setenv("NATIVE_ISAAC_TRAIN_COMMAND", "isaac train --task {task} --manifest {manifest} --config {config}")
    captured: dict[str, object] = {}

    def fake_run_external(*, stage, workspace, command, timeout_seconds, env):
        captured["command"] = tuple(command)
        captured["env"] = dict(env)
        (workspace / "manifest").mkdir(parents=True, exist_ok=True)
        checkpoint = workspace / "checkpoint.pt"
        checkpoint.write_bytes(b"checkpoint")
        return ExternalRunResult(stage, tuple(command), 0, "", "", workspace)

    monkeypatch.setattr("backend.app.runtime.providers.run_external", fake_run_external)
    provider = NativeIsaacLabProvider(registry=registry, workspace=tmp_path / "work")
    run_manifest = RunManifest(
        project_id="project-1",
        run_id="run-1",
        attempt_id="attempt-1",
        robot={"robot_id": "unitree_g1_29dof"},
        motion={"asset_version_id": "motion-1"},
        reward_config_sha256="a" * 64,
        training_config_sha256="b" * 64,
    ).freeze().model_dump(mode="json")
    execution = provider.train(
        run_id="run-1",
        task_id="g1_mimic",
        motion_path=tmp_path / "motion.npz",
        config={"ppo": {"max_iterations": 3}, "_run_manifest": run_manifest},
    )

    manifest_path = execution.result.manifest_path.parent.parent / "run_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == run_manifest
    command = captured["command"]
    assert str(manifest_path) in command
    assert str(manifest_path) != next(item for item in command if item.endswith("training_config.json"))
    assert captured["env"]["ALLROBOTRL_MANIFEST"] == str(manifest_path)
