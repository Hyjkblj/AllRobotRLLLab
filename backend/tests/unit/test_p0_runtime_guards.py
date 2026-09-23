import re
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from adapters.unitree_g1_29dof import UnitreeG1Adapter
from backend.app.application.motion_pipeline_service import MotionPipelineService, MotionPipelineStore
from backend.app.application.training_service import TrainingServiceError
from backend.app.api import routes
from backend.app.config.settings import settings
from backend.app.domain.contracts import TrainingConfig
from backend.app.domain.motion import MotionArrays
from backend.app.runtime.profiles import runtime_names


def _service_block(document: str, service: str) -> str:
    match = re.search(rf"(?ms)^  {re.escape(service)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)", document)
    assert match, f"service {service!r} is missing"
    return match.group("body")


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/runs/h1-run/train",
            "headers": [(b"x-user-id", b"h1-owner")],
        }
    )


def test_api_training_guard_rejects_provider_scoped_to_another_robot(monkeypatch) -> None:
    original_env, original_backend = settings.app_env, settings.p3_backend
    calls: dict[str, object] = {}

    class ProviderService:
        def training_provider_for_run(self, *, run_id, config, actor):
            calls.update(run_id=run_id, task_id=config.task_id, robot_id="unitree_h1_19dof", user_id=actor.user_id)
            raise TrainingServiceError(
                "TRAINING_PROVIDER_ROBOT_UNSUPPORTED",
                "training provider unitree_rl_lab does not support unitree_h1_19dof/h1_mimic",
                status_code=422,
            )

    try:
        settings.app_env = "staging"
        settings.p3_backend = "unitree_rl_lab"
        monkeypatch.setattr(routes, "training_service", ProviderService())
        config = TrainingConfig(task_id="h1_mimic", scene_id="h1_flat", motion_asset_version_id="motion-1")
        with pytest.raises(HTTPException) as raised:
            routes._require_real_backend(_request(), "train", "h1-run", config=config)
        assert raised.value.status_code == 422
        assert raised.value.detail["error"]["code"] == "TRAINING_PROVIDER_ROBOT_UNSUPPORTED"
        assert calls == {"run_id": "h1-run", "task_id": "h1_mimic", "robot_id": "unitree_h1_19dof", "user_id": "h1-owner"}
    finally:
        settings.app_env, settings.p3_backend = original_env, original_backend


def test_api_training_guard_accepts_provider_for_selected_robot(monkeypatch) -> None:
    original_env = settings.app_env

    class ProviderService:
        @staticmethod
        def training_provider_for_run(*, run_id, config, actor):
            assert config.task_id == "h1_mimic"
            return object()

    try:
        settings.app_env = "staging"
        monkeypatch.setattr(routes, "training_service", ProviderService())
        routes._require_real_backend(
            _request(),
            "train",
            "h1-run",
            config=TrainingConfig(task_id="h1_mimic", scene_id="h1_flat", motion_asset_version_id="motion-1"),
        )
    finally:
        settings.app_env = original_env


def test_staging_compose_assigns_process_roles_to_the_correct_services() -> None:
    document = Path("infra/compose/docker-compose.staging.yml").read_text(encoding="utf-8")
    assert re.search(r"(?ms)^      PLATFORM_ROLE: api$", _service_block(document, "api"))
    assert re.search(r"(?ms)^      RUNTIME_PROFILE: api$", _service_block(document, "api"))
    assert re.search(r"(?ms)^      PLATFORM_ROLE: worker-cpu$", _service_block(document, "worker-cpu"))
    assert re.search(r"(?ms)^      RUNTIME_PROFILE: motion-cpu$", _service_block(document, "worker-cpu"))
    assert re.search(r"(?ms)^      PLATFORM_ROLE: worker-gpu$", _service_block(document, "worker-gpu"))
    assert re.search(r"(?ms)^      RUNTIME_PROFILE: gpu$", _service_block(document, "worker-gpu"))


def test_staging_compose_shares_runtime_root_across_processes() -> None:
    document = Path("infra/compose/docker-compose.staging.yml").read_text(encoding="utf-8")
    for service in ("api", "worker-cpu", "outbox-dispatcher", "worker-gpu"):
        assert re.search(r"(?ms)^      ROBOTLAB_RUNTIME_DIR: /app/\.runtime$", _service_block(document, service)), service
        assert "staging-runtime:/app/.runtime" in _service_block(document, service), service


def test_staging_compose_uses_native_isaac_commands_without_unitree_runtime() -> None:
    document = Path("infra/compose/docker-compose.staging.yml").read_text(encoding="utf-8")
    worker = _service_block(document, "worker-gpu")
    assert "NATIVE_ISAAC_TRAIN_COMMAND:" in worker
    assert "NATIVE_ISAAC_EXPORT_COMMAND:" in worker
    assert "NATIVE_ISAAC_PLAY_COMMAND:" in worker
    assert "UNITREE_RL_LAB_PATH" not in worker


def test_deployed_motion_profile_requires_the_g1_mjcf(monkeypatch) -> None:
    original = (settings.app_env, settings.platform_role, settings.runtime_profile, settings.g1_mjcf_path)
    try:
        settings.app_env = "staging"
        settings.platform_role = "worker-cpu"
        settings.runtime_profile = "motion-cpu"
        settings.g1_mjcf_path = ""
        assert any("G1_MJCF_PATH" in error for error in settings.deployment_errors())
    finally:
        settings.app_env, settings.platform_role, settings.runtime_profile, settings.g1_mjcf_path = original


def test_deployed_motion_profile_validates_registered_robot_assets(monkeypatch) -> None:
    original = (settings.app_env, settings.platform_role, settings.runtime_profile)
    try:
        settings.app_env = "staging"
        settings.platform_role = "worker-cpu"
        settings.runtime_profile = "motion-cpu"

        class Registry:
            @staticmethod
            def list():
                class Adapter:
                    name = "fixture_robot"

                    @staticmethod
                    def self_check():
                        from backend.app.domain.contracts import ValidationIssue, ValidationResult, ValidationSeverity

                        return ValidationResult.failure(ValidationIssue(code="ROBOT_ASSET_MISSING", message="fixture model is missing", severity=ValidationSeverity.BLOCKING_ERROR), stage="robot_self_check")

                return [Adapter()]

        errors = settings.deployment_errors(robot_registry=Registry())
        assert any("fixture_robot self-check failed" in error for error in errors)
        assert all("G1_MJCF_PATH" not in error for error in errors)
    finally:
        settings.app_env, settings.platform_role, settings.runtime_profile = original


def test_deployed_motion_compile_refuses_approximation(tmp_path: Path) -> None:
    original = settings.app_env
    try:
        settings.app_env = "production"
        service = object.__new__(MotionPipelineService)
        service.store = MotionPipelineStore(tmp_path / "pipelines")
        service.robot_adapter = UnitreeG1Adapter(repository_root=settings.repository_root)
        service.kinematics_compiler = None
        arrays = MotionArrays(
            fps=30.0,
            joint_pos=np.zeros((1, 29), dtype=np.float32),
            root_pos=np.zeros((1, 3), dtype=np.float32),
            root_rot=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            joint_names=tuple(service.robot_adapter.get_spec().joint_names),
        )
        with pytest.raises(Exception) as raised:
            service._compile(arrays, None, None, None)
        assert getattr(raised.value, "code", "") == "KINEMATICS_RUNTIME_UNAVAILABLE"
    finally:
        settings.app_env = original


def test_runtime_profiles_define_operation_requirements() -> None:
    assert runtime_names("api") == ()
    assert runtime_names("motion-gpu") == ("gmr", "gvhmr")
    assert set(runtime_names("gpu")) == {"gmr", "gvhmr", "isaac_lab", "isaac_sim", "unitree_mujoco"}
    assert "unitree_rl_lab" not in runtime_names("gpu")
