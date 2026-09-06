import json
from pathlib import Path

from backend.app.application.manifest_service import load_runtime_versions
from backend.app.runtime.registry import RuntimeRegistry


def test_runtime_manifest_maps_external_identities(tmp_path) -> None:
    path = tmp_path / "runtime-manifest.json"
    path.write_text(json.dumps({
        "host": {"python": "3.11.9"},
        "external": {
            "isaac_lab": {"git_sha": "abc123"},
            "gmr": {"git_sha": "gmr123"},
            "gvhmr": {"git_sha": "gvh123"},
            "unitree_mujoco": {"git_sha": "mujoco123"},
        },
        "packages": {"isaaclab": "0.47.2", "torch": "2.7.0", "mujoco": "3.3.6"},
        "cuda": {"nvidia_smi": "GPU, 4090, 550.54.14, 24564 MiB"},
    }), encoding="utf-8")
    runtime = load_runtime_versions(path)
    assert runtime is not None
    assert runtime.isaac_lab_git == "v2.3.0@abc123"
    assert runtime.gmr_git == "gmr123"
    assert runtime.cuda_driver == "550.54.14"


def test_runtime_doctor_reports_stale_repository_manifest(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[3]
    path = tmp_path / "runtime-manifest.json"
    path.write_text(json.dumps({"repository": {"root": str(repository), "git_sha": "old-revision"}, "external": {}}), encoding="utf-8")
    registry = RuntimeRegistry(manifest_path=path, registration_path=tmp_path / "registrations.json")
    report = registry.doctor(profile="api")
    assert report["status"] == "NOT_READY"
    assert any(item["name"] == "runtime_manifest" for item in report["failures"])
