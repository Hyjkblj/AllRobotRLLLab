"""Build platform-side handles for independent external runtimes."""

from __future__ import annotations

from pathlib import Path

from backend.app.config.settings import Settings
from backend.app.runtime.gmr_runner import GmrRunner
from backend.app.runtime.gvhmr_runner import GVHMRRunner
from backend.app.runtime.isaac_runner import IsaacLabRunner
from backend.app.runtime.registry import RuntimeRegistry
from backend.app.runtime.unitree_sim2sim_runner import UnitreeMuJoCoRunner
from backend.app.runtime.mujoco_runner import MuJoCoRunner
from backend.app.runtime.providers import NativeIsaacLabProvider, UnitreeRLLabProvider
from backend.app.application.provider_catalog import Sim2SimRegistry, TrainingProviderRegistry


def build_runtime_adapters(settings: Settings, *, workspace: Path, robot_registry=None):
    registry = RuntimeRegistry(manifest_path=settings.runtime_manifest_path, registration_path=settings.runtime_root / "runtime-registrations.json")
    gmr = GmrRunner(registry=registry, workspace=workspace / "gmr")
    gvhmr = GVHMRRunner(registry=registry, workspace=workspace / "gvhmr")
    isaac = IsaacLabRunner(registry=registry, workspace=workspace / "isaac")
    unitree_provider = UnitreeRLLabProvider(isaac)
    native_provider = NativeIsaacLabProvider(registry=registry, workspace=workspace / "native-isaac")
    sim2sim = UnitreeMuJoCoRunner(registry=registry, workspace=workspace / "sim2sim")
    training_provider_registry = TrainingProviderRegistry((unitree_provider, native_provider))
    sim2sim_values: list[object] = [sim2sim]
    if robot_registry is not None:
        for adapter in robot_registry.list():
            spec = adapter.get_spec()
            if spec.sim2sim_adapter == sim2sim.name:
                continue
            if spec.sim2sim_adapter.startswith("mujoco_"):
                command_env = getattr(adapter, "sim2sim_command_env", None) or f"{spec.robot_id.upper()}_SIM2SIM_COMMAND"
                sim2sim_values.append(MuJoCoRunner(name=spec.sim2sim_adapter, workspace=workspace / "sim2sim" / spec.robot_id, command_env=command_env))
    sim2sim_registry = Sim2SimRegistry(sim2sim_values)
    compilers: dict[str, object] = {}
    if robot_registry is not None:
        for adapter in robot_registry.list():
            factory = getattr(adapter, "create_kinematics_compiler", None)
            if not callable(factory):
                continue
            try:
                compiler = factory(allow_approximation=not settings.is_deployed)
            except (OSError, ValueError):
                continue
            compilers[adapter.name] = compiler
    # Compatibility callers may still build runtimes without a registry. In
    # that case no model compiler is created; deployed motion workers will
    # fail closed instead of using an approximation.
    compiler = next(iter(compilers.values()), None)
    return {
        "registry": registry,
        "gmr": gmr,
        "gvhmr": gvhmr,
        # ``isaac`` remains a compatibility alias for the legacy runner.
        "isaac": isaac,
        "providers": {"unitree_rl_lab": unitree_provider, "native_isaac_lab": native_provider},
        "training_provider_registry": training_provider_registry,
        "native_isaac": native_provider,
        "sim2sim": sim2sim,
        "sim2sim_adapters": {item.name: item for item in sim2sim_values},
        "sim2sim_registry": sim2sim_registry,
        "compiler": compiler,
        "kinematics_compilers": compilers,
    }


__all__ = ["build_runtime_adapters"]
