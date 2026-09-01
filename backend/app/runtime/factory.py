"""Build platform-side handles for independent external runtimes."""

from __future__ import annotations

from pathlib import Path

from backend.app.config.settings import Settings
from backend.app.runtime.gmr_runner import GmrRunner
from backend.app.runtime.gvhmr_runner import GVHMRRunner
from backend.app.runtime.isaac_runner import IsaacLabRunner
from backend.app.runtime.mujoco_kinematics import MuJoCoKinematicsCompiler
from backend.app.runtime.registry import RuntimeRegistry
from backend.app.runtime.unitree_sim2sim_runner import UnitreeMuJoCoRunner


def build_runtime_adapters(settings: Settings, *, workspace: Path):
    registry = RuntimeRegistry(manifest_path=settings.runtime_manifest_path, registration_path=settings.runtime_root / "runtime-registrations.json")
    gmr = GmrRunner(registry=registry, workspace=workspace / "gmr")
    gvhmr = GVHMRRunner(registry=registry, workspace=workspace / "gvhmr")
    isaac = IsaacLabRunner(registry=registry, workspace=workspace / "isaac")
    sim2sim = UnitreeMuJoCoRunner(registry=registry, workspace=workspace / "sim2sim")
    compiler = None
    if settings.g1_mjcf_path:
        from adapters.unitree_g1_29dof import UnitreeG1Adapter

        spec = UnitreeG1Adapter(repository_root=settings.repository_root).get_spec()
        compiler = MuJoCoKinematicsCompiler(model_path=Path(settings.g1_mjcf_path), body_names=spec.body_names, joint_names=spec.joint_names, allow_approximation=not settings.is_deployed)
    return {"registry": registry, "gmr": gmr, "gvhmr": gvhmr, "isaac": isaac, "sim2sim": sim2sim, "compiler": compiler}


__all__ = ["build_runtime_adapters"]
