"""GVHMR process adapter."""

from __future__ import annotations

from pathlib import Path

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
        result = run_external(stage="gvhmr", workspace=target, command=command or default, timeout_seconds=self.timeout_seconds, env={"GVHMR_INPUT": str(source), "GVHMR_OUTPUT": str(target)})
        candidates = [target / "hmr4d_results.pt", *sorted(target.rglob("hmr4d_results.pt"))]
        output = next((path for path in candidates if path.is_file()), None)
        if output is None:
            raise RunnerError("RUNTIME_OUTPUT_MISSING", "GVHMR completed without hmr4d_results.pt", details={"workspace": str(target)})
        manifest = write_output_manifest(target, stage="gvhmr", outputs=[output], metadata={"runtime": check.as_dict(), "adapter_version": self.version})
        return output, ExternalRunResult(result.stage, result.command, result.return_code, result.stdout, result.stderr, result.workspace, {"gvhmr_result": output}, manifest)


GvhMrRunner = GVHMRRunner

__all__ = ["GVHMRRunner", "GvhMrRunner"]
