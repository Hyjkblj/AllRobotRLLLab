"""Policy export, CPU smoke validation, and bundle assembly."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from backend.app.domain.contracts import ExportFile, ExportMetadata, PolicyBundle, Sim2SimReport


class ExportError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(root: Path) -> list[str]:
    """Validate the bundle checksum manifest and return any diagnostics."""
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        return ["CHECKSUM_MANIFEST_MISSING"]
    failures: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="ascii").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"CHECKSUM_LINE_INVALID_{line_number}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            failures.append(f"CHECKSUM_PATH_OUTSIDE_ROOT_{line_number}")
            continue
        if relative in seen or relative == "checksums.sha256":
            failures.append(f"CHECKSUM_PATH_DUPLICATE_{line_number}")
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append(f"CHECKSUM_FILE_MISSING_{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"CHECKSUM_MISMATCH_{relative}")
    expected_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "checksums.sha256"}
    failures.extend(f"CHECKSUM_FILE_UNLISTED_{relative}" for relative in sorted(expected_files - seen))
    return failures


def file_record(path: Path, *, root: Path, format: str) -> ExportFile:
    return ExportFile(path=str(path.resolve().relative_to(root.resolve())).replace("\\", "/"), format=format, sha256=sha256_file(path), size_bytes=path.stat().st_size)


@dataclass(frozen=True)
class ExportResult:
    metadata: ExportMetadata
    files: list[ExportFile]
    output_dir: Path


class TorchPolicyExporter:
    """Export a feed-forward policy when the Isaac/Torch runtime is present."""

    def export(self, *, output_dir: Path, input_dim: int, output_dim: int, action_scale: float, opset: int = 17) -> ExportResult:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - server image only
            raise ExportError("EXPORT_RUNTIME_UNAVAILABLE", "torch is not installed in the selected training environment") from exc
        output_dir.mkdir(parents=True, exist_ok=True)
        class Policy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, 128), torch.nn.Tanh(), torch.nn.Linear(128, output_dim), torch.nn.Tanh())

            def forward(self, observation):
                return self.net(observation) * action_scale

        model = Policy().eval()
        dummy = torch.zeros((2, input_dim), dtype=torch.float32)
        jit_path = output_dir / "policy.pt"
        onnx_path = output_dir / "policy.onnx"
        try:
            scripted = torch.jit.trace(model, dummy)
            scripted.save(str(jit_path))
            torch.onnx.export(model, dummy, str(onnx_path), opset_version=opset, input_names=["observation"], output_names=["action"], dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}}, do_constant_folding=True)
            jit_model = torch.jit.load(str(jit_path), map_location="cpu").eval()
            result = jit_model(dummy)
            if tuple(result.shape) != (2, output_dim) or not bool(torch.isfinite(result).all()):
                raise ExportError("EXPORT_SMOKE_FAILED", "TorchScript output shape or finite check failed")
            try:
                import onnxruntime as ort
                import numpy as np

                session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                onnx_result = session.run(["action"], {"observation": np.zeros((2, input_dim), dtype=np.float32)})[0]
                if tuple(onnx_result.shape) != (2, output_dim) or not np.isfinite(onnx_result).all():
                    raise ExportError("EXPORT_SMOKE_FAILED", "ONNX output shape or finite check failed")
            except ExportError:
                raise
            except ImportError as exc:
                raise ExportError("EXPORT_RUNTIME_UNAVAILABLE", "onnxruntime is not installed for CPU smoke inference") from exc
        except ExportError:
            raise
        except Exception as exc:
            raise ExportError("EXPORT_INVALID", f"policy export failed: {exc}") from exc
        files = [file_record(jit_path, root=output_dir, format="torchscript"), file_record(onnx_path, root=output_dir, format="onnx")]
        metadata = ExportMetadata(policy_input_dim=input_dim, policy_output_dim=output_dim, action_scale=action_scale, onnx_opset=opset, input_name="observation", output_name="action", exporter="torch-policy-exporter.v1", runtime=f"torch.{torch.__version__}", smoke_passed=True, files=files)
        (output_dir / "export_meta.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        files.append(file_record(output_dir / "export_meta.json", root=output_dir, format="json"))
        return ExportResult(metadata=metadata, files=files, output_dir=output_dir)


def build_policy_bundle(*, output_dir: Path, run_id: str, attempt_id: str, robot_id: str, observation_dim: int, action_dim: int, export: ExportResult, manifest: dict, sim2sim_report: dict | None = None) -> PolicyBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    for exported_file in export.files:
        source = export.output_dir / exported_file.path
        target = output_dir / exported_file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    params_dir = output_dir / "params"
    manifest_dir = output_dir / "manifest"
    params_dir.mkdir(exist_ok=True)
    manifest_dir.mkdir(exist_ok=True)
    (params_dir / "deploy.yaml").write_text(f"robot_id: {robot_id}\ncontrol_dt: 0.02\naction_dim: {action_dim}\n", encoding="utf-8")
    (params_dir / "env.yaml").write_text(f"observation_dim: {observation_dim}\nscene_id: g1_flat\n", encoding="utf-8")
    (params_dir / "agent.yaml").write_text("algorithm: rsl_rl_ppo\n", encoding="utf-8")
    (params_dir / "normalization.json").write_text(json.dumps({"mean": [0.0] * observation_dim, "std": [1.0] * observation_dim}, separators=(",", ":")), encoding="utf-8")
    (params_dir / "action_scale.json").write_text(json.dumps({"scale": export.metadata.action_scale}), encoding="utf-8")
    (manifest_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    (manifest_dir / "manifest.sha256").write_text(hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(), encoding="ascii")
    if sim2sim_report is not None:
        (output_dir / "sim2sim_report.json").write_text(json.dumps(sim2sim_report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    content_files = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"]
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in sorted(content_files)]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    checksum_errors = verify_checksums(output_dir)
    if checksum_errors:
        raise ExportError("BUNDLE_CHECKSUM_INVALID", "; ".join(checksum_errors))
    all_files = [path for path in output_dir.rglob("*") if path.is_file()]
    archive_path = output_dir.parent / "policy_bundle.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(output_dir, arcname="policy_bundle")
    bundle_files = [file_record(path, root=output_dir, format=path.suffix.lstrip(".") or "file") for path in all_files]
    bundle_files.append(ExportFile(path=archive_path.name, format="tar.gz", sha256=sha256_file(archive_path), size_bytes=archive_path.stat().st_size))
    report_model = Sim2SimReport.model_validate(sim2sim_report) if sim2sim_report is not None else None
    bundle = PolicyBundle(bundle_id=str(uuid.uuid4()), run_id=run_id, attempt_id=attempt_id, robot_id=robot_id, observation_dim=observation_dim, action_dim=action_dim, files=bundle_files, export=export.metadata, sim2sim_report=report_model, status="EXPORTED")
    return bundle


__all__ = ["ExportError", "ExportResult", "TorchPolicyExporter", "build_policy_bundle", "file_record", "sha256_file", "verify_checksums"]
