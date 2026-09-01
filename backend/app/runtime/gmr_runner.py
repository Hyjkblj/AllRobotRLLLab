"""GMR retargeting adapter and output normalizer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.runtime.contracts import ExternalRunResult, RunnerError
from backend.app.runtime.process import command_from_env, run_external, write_output_manifest
from backend.app.runtime.registry import RuntimeRegistry


def _as_array(payload: dict[str, Any], names: tuple[str, ...], *, ndim: int = 2) -> np.ndarray:
    key = next((name for name in names if name in payload), None)
    if key is None:
        raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", f"GMR output is missing one of {names}")
    value = np.asarray(payload[key], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", f"GMR field {key} must be finite with rank {ndim}")
    return value


class GmrRunner:
    name = "gmr"
    version = "gmr-adapter.v1"

    def __init__(self, *, registry: RuntimeRegistry, workspace: Path, timeout_seconds: float = 3600) -> None:
        self.registry = registry
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def run(self, *, source_path: Path, output_dir: Path | None = None, robot: str = "unitree_g1") -> tuple[Path, ExternalRunResult]:
        check = self.registry.require("gmr")
        source = Path(source_path).resolve()
        if not source.is_file():
            raise RunnerError("INPUT_NOT_FOUND", f"GMR source does not exist: {source}")
        target = Path(output_dir or self.workspace / "gmr").resolve()
        target.mkdir(parents=True, exist_ok=True)
        output = target / "retarget_motion.pkl"
        if source.suffix.lower() == ".pt":
            script = Path(check.path or ".") / "scripts" / "gvhmr_to_robot.py"
            default = [check.python or "python", str(script), "--gvhmr_pred_file", str(source), "--robot", robot, "--save_path", str(output)]
        elif source.suffix.lower() == ".bvh":
            script = Path(check.path or ".") / "scripts" / "bvh_to_robot.py"
            default = [check.python or "python", str(script), "--bvh_file", str(source), "--robot", robot, "--save_path", str(output)]
        else:
            raise RunnerError("GMR_INPUT_UNSUPPORTED", f"GMR accepts GVHMR .pt or BVH input, got {source.suffix}")
        command = command_from_env("GMR_COMMAND", default=default)
        if command and any("{input}" in item or "{output}" in item for item in command):
            command = tuple(item.format(input=str(source), output=str(output), robot=robot) for item in command)
        result = run_external(stage="gmr", workspace=target, command=command or default, timeout_seconds=self.timeout_seconds, env={"GMR_INPUT": str(source), "GMR_OUTPUT": str(output), "GMR_ROBOT": robot})
        if not output.is_file():
            candidates = sorted(target.glob("*.pkl"))
            output = candidates[0] if candidates else None
        if output is None or not output.is_file():
            raise RunnerError("RUNTIME_OUTPUT_MISSING", "GMR completed without a retarget output")
        normalized = self._validate_and_normalize(output, target)
        manifest = write_output_manifest(target, stage="gmr", outputs=[output, normalized], metadata={"runtime": check.as_dict(), "adapter_version": self.version, "robot": robot})
        return normalized, ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {"retarget_motion": normalized, "source_pickle": output}, manifest)

    @staticmethod
    def _validate_and_normalize(path: Path, root: Path) -> Path:
        try:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", f"unable to read GMR output: {exc}") from exc
        if not isinstance(payload, dict):
            raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", "GMR output must be a dictionary")
        dof = _as_array(payload, ("dof_pos", "joint_pos", "qpos"))
        if dof.shape[1] not in (29, 36):
            raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", f"G1 output must contain 29 DoF, got {dof.shape[1]}")
        root_pos = _as_array(payload, ("root_pos", "base_pos"))
        root_rot = _as_array(payload, ("root_rot", "base_rot"))
        if root_pos.shape != (dof.shape[0], 3) or root_rot.shape != (dof.shape[0], 4):
            raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", "root arrays do not match DoF frame count")
        fps = float(payload.get("fps", 30.0))
        if not 15 <= fps <= 120:
            raise RunnerError("GMR_OUTPUT_SCHEMA_INVALID", "GMR fps must be between 15 and 120")
        # Keep a stable, safe-to-consume NPZ alongside the original pickle.
        normalized = root / "retarget_motion.npz"
        np.savez_compressed(normalized, joint_pos=dof[:, :29], root_pos=root_pos, root_rot=root_rot, fps=np.asarray(fps, dtype=np.float32), joint_names=np.asarray(payload.get("joint_names", [])))
        return normalized


__all__ = ["GmrRunner"]
