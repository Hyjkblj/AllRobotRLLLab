"""Manifest freeze service."""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend.app.domain.contracts import RunManifest, RuntimeVersions


def freeze_manifest(manifest: RunManifest) -> RunManifest:
    """Return an immutable-by-contract manifest with its canonical hash pinned."""

    return manifest.freeze()


def load_runtime_versions(path: Path) -> RuntimeVersions | None:
    """Load host identities collected by ``collect_runtime_manifest.py``."""
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        external = document.get("external", {})
        packages = document.get("packages", {})
        cuda = document.get("cuda", {})
        nvidia = str(cuda.get("nvidia_smi") or "")
        driver = nvidia.split(",", 3)[2].strip() if nvidia.count(",") >= 2 else "pending"
        values = {
            "isaac_lab_git": f"v2.3.0@{external.get('isaac_lab', {}).get('git_sha') or 'pending'}",
            "isaac_lab_package": packages.get("isaaclab") or "pending",
            "isaac_sim_package": os.getenv("ISAACSIM_VERSION", "5.1.0.0"),
            "unitree_rl_lab_package": packages.get("unitree-rl-lab") or "0.2.1",
            "unitree_mujoco_git": external.get("unitree_mujoco", {}).get("git_sha") or (f"content:{external.get('unitree_mujoco', {}).get('source_sha256')}" if external.get("unitree_mujoco", {}).get("source_sha256") else "pending"),
            "gmr_git": external.get("gmr", {}).get("git_sha") or (f"content:{external.get('gmr', {}).get('source_sha256')}" if external.get("gmr", {}).get("source_sha256") else "pending"),
            "gvhmr_git": external.get("gvhmr", {}).get("git_sha") or (f"content:{external.get('gvhmr', {}).get('source_sha256')}" if external.get("gvhmr", {}).get("source_sha256") else "pending"),
            "mujoco_runtime": packages.get("mujoco") or "pending",
            "python": document.get("host", {}).get("python") or "3.11",
            "torch": packages.get("torch") or "pending",
            "cuda_driver": driver,
            "container_digest": os.getenv("CONTAINER_DIGEST", "pending"),
        }
        return RuntimeVersions.model_validate(values)
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return None


__all__ = ["freeze_manifest", "load_runtime_versions"]
