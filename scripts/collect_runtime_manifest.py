"""Collect immutable identities for a server or GPU acceptance run.

The script only reads the host and external runtime directories.  It never
downloads or installs third-party software and writes a small JSON manifest
that can be attached to a Run or retained with deployment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PINNED_PATHS = {
    "ISAACLAB_PATH": "isaac_lab",
    "ISAACSIM_PATH": "isaac_sim",
    "GMR_PATH": "gmr",
    "GVHMR_PATH": "gvhmr",
    "UNITREE_MUJOCO_PATH": "unitree_mujoco",
    "UNITREE_RL_LAB_PATH": "unitree_rl_lab",
}


def _discover_isaacsim() -> Path | None:
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin).resolve()
    return origin.parent.parent if origin.parent.name == "isaacsim" else origin.parent


def _command(*args: str) -> str | None:
    if shutil.which(args[0]) is None:
        return None
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=10)
    value = (result.stdout or result.stderr).strip()
    return value or None


def _git_revision(path: Path) -> str | None:
    root = _command("git", "-C", str(path), "rev-parse", "--show-toplevel")
    if not root:
        return None
    try:
        if Path(root).resolve() != path.resolve():
            return None
    except OSError:
        return None
    return _command("git", "-C", str(path), "rev-parse", "HEAD")


def _source_hash(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file() and ".git" not in item.parts):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_identity(root: Path) -> dict[str, Any]:
    gmr_root = os.getenv("GMR_PATH", "").strip()
    default_gmr = Path(gmr_root).expanduser() if gmr_root else root / "third_party" / "GMR-master"
    candidates = {
        "mujoco_xml": os.getenv("G1_MJCF_PATH", str(default_gmr / "assets" / "unitree_g1" / "g1_mocap_29dof.xml")),
        "urdf": os.getenv("G1_URDF_PATH", str(default_gmr / "assets" / "unitree_g1" / "g1_custom_collision_29dof.urdf")),
        # Unitree RL Lab's G1 29 DoF mimic task imports this URDF and lets
        # Isaac Lab generate the runtime USD representation. Keep it separate
        # from the GMR URDF used by the motion/contract adapter.
        "isaac_urdf": os.getenv("G1_ISAAC_URDF_PATH", ""),
        "isaac_usd": os.getenv("G1_USD_PATH", ""),
    }
    result: dict[str, Any] = {}
    for name, raw in candidates.items():
        if not raw:
            result[name] = {"path": None, "status": "not_configured"}
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_file():
            result[name] = {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        else:
            result[name] = {"path": str(path), "status": "missing"}
    return result


def collect(root: Path) -> dict[str, Any]:
    registration_file = root / ".runtime" / "runtime-registrations.json"
    try:
        value = json.loads(registration_file.read_text(encoding="utf-8")) if registration_file.is_file() else {}
        registrations = value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        registrations = {}
    external: dict[str, Any] = {}
    for variable, name in PINNED_PATHS.items():
        registration = registrations.get(name, {}) if isinstance(registrations.get(name, {}), dict) else {}
        raw = os.getenv(variable, "").strip() or str(registration.get("path", "")).strip()
        if variable == "ISAACSIM_PATH" and not raw:
            discovered = _discover_isaacsim()
            if discovered is not None:
                raw = str(discovered)
        if not raw:
            external[name] = {"env": variable, "status": "not_configured"}
            continue
        path = Path(raw).expanduser().resolve()
        external[name] = {
            "env": variable,
            "path": str(path),
            "exists": path.is_dir(),
            "git_sha": _git_revision(path) if path.is_dir() else None,
            "source_sha256": _source_hash(path) if path.is_dir() else None,
        }
    packages: dict[str, str | None] = {}
    for package in ("torch", "isaaclab", "unitree-rl-lab", "mujoco", "celery", "fastapi"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpus = _command("nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader")
    return {
        "schema_version": "runtime_manifest.v1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0]},
        "repository": {"root": str(root), "git_sha": _git_revision(root)},
        "external": external,
        "packages": packages,
        "cuda": {"nvidia_smi": gpus, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
        "assets": _asset_identity(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--output", type=Path, default=None, help="JSON destination; defaults to .runtime/runtime-manifest.json")
    parser.add_argument("--strict", action="store_true", help="fail when configured external paths are missing")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = collect(root)
    output = (args.output or root / ".runtime" / "runtime-manifest.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "repository_sha": manifest["repository"]["git_sha"], "assets": manifest["assets"]}, ensure_ascii=False))
    if args.strict:
        missing = [name for name, item in manifest["external"].items() if item.get("status") == "not_configured" or item.get("exists") is False]
        missing.extend(name for name, item in manifest["assets"].items() if item.get("status") == "missing")
        if missing:
            print("Missing runtime identities: " + ", ".join(sorted(missing)), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
