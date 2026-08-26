import json

from backend.app.application.manifest_service import load_runtime_versions


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
