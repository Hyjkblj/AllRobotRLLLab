"""GVHMR process adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.runtime.contracts import ExternalRunResult, RunnerError
from backend.app.runtime.process import command_from_env, run_external, write_output_manifest
from backend.app.runtime.registry import RuntimeRegistry


class GVHMRRunner:
    """Run GVHMR in its own Python/CUDA environment.

    The command can be overridden with ``GVHMR_COMMAND``.  Without an
    override the upstream ``tools/demo/demo.py`` entry point is used.
    """

    name = "gvhmr"
    version = "gvhmr-adapter.v1"

    def __init__(self, *, registry: RuntimeRegistry, workspace: Path, timeout_seconds: float = 3600) -> None:
        self.registry = registry
        self.workspace = Path(workspace).resolve()
        self.timeout_seconds = timeout_seconds

    def run(self, *, video_path: Path, output_dir: Path | None = None, static_cam: bool = True) -> tuple[Path, ExternalRunResult]:
        check = self.registry.require("gvhmr")
        source = Path(video_path).resolve()
        if not source.is_file():
            raise RunnerError("INPUT_NOT_FOUND", f"video does not exist: {source}")
        target = Path(output_dir or self.workspace / "gvhmr").resolve()
        target.mkdir(parents=True, exist_ok=True)
        default = [check.python or "python", str(Path(check.path or ".") / "tools" / "demo" / "demo.py"), f"--video={source}", f"--output_root={target}"]
        if static_cam:
            default.append("-s")
        command = command_from_env("GVHMR_COMMAND", default=default)
        # Explicit commands are templates so deployments can pin an entry
        # point while still receiving the same input/output contract.
        if command and "{input}" in command:
            command = tuple(item.format(input=str(source), output=str(target)) for item in command)
        # GVHMR resolves model/checkpoint paths such as
        # ``inputs/checkpoints/vitpose/...`` relative to its repository root.
        # Keep artifacts in the pipeline workspace while running the external
        # process from the registered GVHMR checkout.
        result = run_external(stage="gvhmr", workspace=target, command=command or default, timeout_seconds=self.timeout_seconds, cwd=Path(check.path or "."), env={"GVHMR_INPUT": str(source), "GVHMR_OUTPUT": str(target)})
        candidates = [target / "hmr4d_results.pt", *sorted(target.rglob("hmr4d_results.pt"))]
        output = next((path for path in candidates if path.is_file()), None)
        if output is None:
            raise RunnerError("RUNTIME_OUTPUT_MISSING", "GVHMR completed without hmr4d_results.pt", details={"workspace": str(target)})
        manifest = write_output_manifest(target, stage="gvhmr", outputs=[output], metadata={"runtime": check.as_dict(), "adapter_version": self.version})
        return output, ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {"gvhmr_result": output}, manifest)

    @staticmethod
    def validate_output(path: Path) -> dict[str, Any]:
        """Validate GVHMR's serialized output without running GMR.

        This is deliberately a format/schema gate only.  It does not judge
        pose quality; that belongs to the later robot-retargeting checks.
        """
        source = Path(path).resolve()
        if not source.is_file():
            raise RunnerError("GVHMR_OUTPUT_NOT_FOUND", f"GVHMR output does not exist: {source}")
        try:
            import torch
            try:
                payload = torch.load(source, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(source, map_location="cpu")
        except Exception as exc:
            raise RunnerError("GVHMR_OUTPUT_INVALID", f"unable to load GVHMR output: {exc}") from exc
        if not isinstance(payload, dict):
            raise RunnerError("GVHMR_OUTPUT_SCHEMA_INVALID", "GVHMR output must be a dictionary")
        params = payload.get("smpl_params_global") or payload.get("smpl_params")
        if not isinstance(params, dict):
            raise RunnerError("GVHMR_OUTPUT_SCHEMA_INVALID", "GVHMR output is missing smpl_params_global")
        required = ("body_pose", "global_orient", "transl")
        missing = [name for name in required if name not in params]
        if missing:
            raise RunnerError("GVHMR_OUTPUT_SCHEMA_INVALID", f"GVHMR output is missing fields: {', '.join(missing)}")
        frames = None
        for name in required:
            value = params[name]
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) < 2 or int(shape[0]) <= 0:
                raise RunnerError("GVHMR_OUTPUT_SCHEMA_INVALID", f"GVHMR field {name} must contain frame data")
            current = int(shape[0])
            frames = current if frames is None else frames
            if current != frames:
                raise RunnerError("GVHMR_OUTPUT_SCHEMA_INVALID", "GVHMR fields have inconsistent frame counts")
        return {"format": "gvhmr_hmr4d", "frame_count": frames, "fields": list(required)}


GvhMrRunner = GVHMRRunner

__all__ = ["GVHMRRunner", "GvhMrRunner"]
