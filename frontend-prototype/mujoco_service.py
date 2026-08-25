"""MuJoCo-backed Motion Lab service.

The service deliberately resolves robot assets from this repository's
``third_party`` directory. Motion assets are discovered from the local
UnitreeG1Dance workspace and loaded lazily.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import struct
import sys
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# The web service renders off-screen with MuJoCo's real renderer. Select the
# backend before importing mujoco: EGL is the preferred Linux/GPU backend,
# OSMesa is the CPU fallback, and GLFW is used for local Windows development.
RENDER_BACKEND = os.environ.get("MOTIONLAB_RENDER_BACKEND") or os.environ.get("MUJOCO_GL")
if RENDER_BACKEND:
    os.environ["MUJOCO_GL"] = RENDER_BACKEND
elif os.name == "nt":
    RENDER_BACKEND = "glfw"
    os.environ["MUJOCO_GL"] = RENDER_BACKEND
else:
    # MuJoCo's Linux wheel can use EGL when a display is not available. Keep
    # this explicit so a headless deployment does not accidentally seek X11.
    RENDER_BACKEND = "egl"
    os.environ["MUJOCO_GL"] = RENDER_BACKEND

try:
    import numpy as np
except ImportError:  # pragma: no cover - the conda bootstrap installs numpy
    np = None

import mujoco

try:  # torch is optional; qpos .npz/.csv assets work without it.
    import torch
except ImportError:  # pragma: no cover - depends on the selected conda env
    torch = None


ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
ACTION_ROOT = Path(
    os.environ.get("MOTIONLAB_ACTION_ROOT", r"D:\Develop\Project\UnitreeG1Dance")
).expanduser()
MODEL_OVERRIDE = os.environ.get("MOTIONLAB_MODEL_PATH")
URDF_OVERRIDE = os.environ.get("MOTIONLAB_URDF_PATH")
MODEL_CANDIDATES = [
    Path(MODEL_OVERRIDE) if MODEL_OVERRIDE else None,
    THIRD_PARTY / "GMR-master" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
    ACTION_ROOT / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
]
URDF_CANDIDATES = [
    Path(URDF_OVERRIDE) if URDF_OVERRIDE else None,
    THIRD_PARTY / "GMR-master" / "assets" / "unitree_g1" / "g1_custom_collision_29dof.urdf",
    ACTION_ROOT / "GMR" / "assets" / "unitree_g1" / "g1_custom_collision_29dof.urdf",
]
SOURCE_MODEL_PATH = next((p for p in MODEL_CANDIDATES if p and p.exists()), None)
URDF_PATH = next((p for p in URDF_CANDIDATES if p and p.exists()), None)
if SOURCE_MODEL_PATH is None:
    raise FileNotFoundError(
        "G1 MuJoCo model was not found in third_party/GMR-master/assets/unitree_g1"
    )


RUNTIME_MODEL_ROOT = ROOT / "frontend-prototype" / ".runtime" / "g1_mocap_29dof"


def _prepare_runtime_model(source: Path) -> Path:
    """Stage the repository MJCF and meshes for MuJoCo's Windows STL decoder.

    The GMR assets are binary STL files with a textual ``solid`` prefix. Some
    MuJoCo Windows wheels classify those files as ASCII and reject them. The
    mesh bytes are unchanged; only the five-byte STL header is made unambiguous
    in a generated runtime cache outside ``third_party``.
    """
    target = RUNTIME_MODEL_ROOT / source.name
    stamp = RUNTIME_MODEL_ROOT / ".source-stamp"
    source_stamp = str(source.stat().st_mtime_ns)
    if target.exists() and stamp.exists() and stamp.read_text(encoding="ascii") == source_stamp:
        return target
    if RUNTIME_MODEL_ROOT.exists():
        shutil.rmtree(RUNTIME_MODEL_ROOT, ignore_errors=True)
    (RUNTIME_MODEL_ROOT / "meshes").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    xml_text = source.read_text(encoding="utf-8")
    for mesh_name in re.findall(r'file="([^"]+)"', xml_text):
        source_mesh = source.parent / "meshes" / mesh_name
        target_mesh = RUNTIME_MODEL_ROOT / "meshes" / mesh_name
        if not source_mesh.exists():
            continue
        shutil.copy2(source_mesh, target_mesh)
        raw = bytearray(target_mesh.read_bytes())
        if len(raw) >= 84 and raw[:5].lower() == b"solid":
            triangles = struct.unpack_from("<I", raw, 80)[0]
            if 84 + triangles * 50 <= len(raw):
                raw[:5] = b"G1BIN"
                target_mesh.write_bytes(raw)
    stamp.write_text(source_stamp, encoding="ascii")
    return target


MODEL_PATH = _prepare_runtime_model(SOURCE_MODEL_PATH)

FPS = 30
FRAME_COUNT = 121
RUNTIME_VERSION = str(getattr(mujoco, "__version__", "unknown"))
MUJOCO_SOURCE_VERSION = "3.12.1"  # third_party/mujoco-main/CMakeLists.txt


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def jsonable(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


class ActionAsset:
    def __init__(self, path: Path, root: Path):
        self.path = path.resolve()
        self.root = root.resolve()
        self.relative_path = str(self.path.relative_to(self.root)).replace("\\", "/")
        self.asset_id = hashlib.sha1(self.relative_path.encode("utf-8")).hexdigest()[:12]
        self.suffix = self.path.suffix.lower()
        self.kind = "policy" if self.suffix == ".pt" else "motion"
        if self.suffix == ".pkl":
            self.kind = "source_motion"
        self.size = self.path.stat().st_size
        self.modified_at = datetime.fromtimestamp(self.path.stat().st_mtime, timezone.utc).isoformat()
        self.fps = FPS
        self.frame_count = 0
        self.nq = None
        self.joint_names: list[str] = []
        self.load_error: str | None = None
        self._frames: list[list[float]] | None = None

    def public(self) -> dict:
        return {
            "id": self.asset_id,
            "name": self.path.stem,
            "fileName": self.path.name,
            "relativePath": self.relative_path,
            "absolutePath": str(self.path),
            "extension": self.suffix.lstrip("."),
            "kind": self.kind,
            "sizeBytes": self.size,
            "modifiedAt": self.modified_at,
            "fps": self.fps,
            "frameCount": self.frame_count,
            "nq": self.nq,
            "jointNames": self.joint_names,
            "loadError": self.load_error,
            "source": "UnitreeG1Dance",
        }


class ActionLibrary:
    """Discover and lazily decode local motion/policy artifacts."""

    EXTENSIONS = {".pt", ".npz", ".csv", ".pkl"}

    def __init__(self, root: Path, model: mujoco.MjModel, joints: list[dict]):
        self.root = root.resolve()
        self.model = model
        self.joints = joints
        self.joint_by_name = {joint["name"]: joint for joint in joints}
        self.assets: dict[str, ActionAsset] = {}
        self._lock = threading.RLock()
        self._scan()

    def _scan(self) -> None:
        if not self.root.exists():
            return
        for path in self.root.rglob("*"):
            try:
                if not path.is_file() or path.suffix.lower() not in self.EXTENSIONS:
                    continue
            except OSError:
                # UnitreeG1Dance contains a few Windows links to external
                # workspaces. They are not required for local action playback.
                continue
            if any(part in {".git", "__pycache__", "node_modules"} for part in path.parts):
                continue
            try:
                asset = ActionAsset(path, self.root)
                if asset.suffix == ".npz":
                    self._inspect_npz(asset)
                elif asset.suffix == ".csv":
                    self._inspect_csv(asset)
                self.assets[asset.asset_id] = asset
            except (OSError, ValueError):
                continue

    def _inspect_npz(self, asset: ActionAsset) -> None:
        if np is None:
            asset.load_error = "numpy is not installed in the active conda environment"
            return
        try:
            with np.load(asset.path, allow_pickle=True) as archive:
                qpos = archive.get("qpos")
                if qpos is not None and getattr(qpos, "ndim", 0) >= 2:
                    asset.frame_count = int(qpos.shape[0])
                    asset.nq = int(qpos.shape[-1])
                if "fps" in archive:
                    asset.fps = int(np.asarray(archive["fps"]).reshape(-1)[0])
                if "joint_names" in archive:
                    asset.joint_names = [str(v) for v in np.asarray(archive["joint_names"]).reshape(-1)]
        except Exception as exc:
            asset.load_error = str(exc)

    def _inspect_csv(self, asset: ActionAsset) -> None:
        try:
            with asset.path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                first = next(reader, [])
                if first and first[0].lstrip().startswith("# joint_names"):
                    asset.joint_names = [name for name in first[1:] if name]
                    header = next(reader, [])
                    data_start = 2 if header[:2] == ["time_s", "phase"] else 0
                else:
                    header = first
                    data_start = 0
                rows = sum(1 for _ in reader)
            asset.frame_count = rows
            asset.nq = max(0, len(header) - data_start)
            if not asset.joint_names:
                asset.joint_names = [name for name in header[data_start:] if name]
        except Exception as exc:
            asset.load_error = str(exc)

    def register(self, path: Path) -> ActionAsset:
        path = path.resolve()
        asset = ActionAsset(path, self.root)
        if asset.suffix == ".npz":
            self._inspect_npz(asset)
        elif asset.suffix == ".csv":
            self._inspect_csv(asset)
        self.assets[asset.asset_id] = asset
        return asset

    def get(self, asset_id: str) -> ActionAsset:
        asset = self.assets.get(asset_id)
        if asset is None:
            raise ValueError(f"unknown action asset: {asset_id}")
        return asset

    def list_public(self, include_source: bool = False) -> list[dict]:
        assets = list(self.assets.values())
        if not include_source:
            assets = [asset for asset in assets if asset.kind != "source_motion"]
        assets.sort(key=lambda asset: (0 if asset.kind == "motion" else 1, asset.relative_path.lower()))
        return [asset.public() for asset in assets]

    def _normalise_array(self, values: Any) -> Any:
        if np is None:
            raise RuntimeError("numpy is not installed in the active conda environment")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        return array

    def _map_array(self, values: Any, names: list[str] | None = None) -> list[list[float]]:
        array = self._normalise_array(values)
        default = np.asarray(self.model.qpos0, dtype=np.float64)
        if len(default) >= 7:
            default[0:7] = [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]
        result = np.repeat(default.reshape(1, -1), array.shape[0], axis=0)
        width = int(array.shape[1])
        names = [str(name) for name in (names or [])]
        if width == self.model.nq:
            result[:, :] = array[:, : self.model.nq]
            # GMR/Unitree reference files store the free-base z as a local
            # offset (usually 0). This MJCF uses a free joint, so MuJoCo treats
            # qpos[2] as world z and would otherwise sink the real mesh below
            # the floor. Convert the reference to the model's standing height.
            if result.shape[1] >= 3 and float(np.max(result[:, 2])) < 0.3:
                free_body = 1 if self.model.nbody > 1 else 0
                result[:, 2] += float(self.model.body_pos[free_body][2])
        elif names and len(names) == width:
            for index, name in enumerate(names):
                joint = self.joint_by_name.get(name)
                if joint:
                    result[:, joint["qposAdr"]] = array[:, index]
        elif width == len(self.joints):
            for index, joint in enumerate(self.joints):
                result[:, joint["qposAdr"]] = array[:, index]
        elif width == self.model.nv and self.model.nv == self.model.nq - 1:
            for index, joint in enumerate(self.joints):
                if index < width:
                    result[:, joint["qposAdr"]] = array[:, index]
        else:
            raise ValueError(
                f"motion tensor has {width} values per frame; expected nq={self.model.nq} or DoF={len(self.joints)}"
            )
        return result.tolist()

    def _extract_tensor(self, value: Any) -> Any:
        if torch is not None and torch.is_tensor(value):
            candidate = value.detach().cpu().numpy()
            if getattr(candidate, "ndim", 0) >= 2 and int(candidate.shape[-1]) in {self.model.nq, self.model.nv, len(self.joints)}:
                return candidate
            return None
        if np is not None and isinstance(value, np.ndarray):
            if value.ndim >= 2 and int(value.shape[-1]) in {self.model.nq, self.model.nv, len(self.joints)}:
                return value
            return None
        if isinstance(value, dict):
            for key in ("qpos", "actions", "motion", "poses", "data"):
                if key in value:
                    found = self._extract_tensor(value[key])
                    if found is not None:
                        return found
            for nested in value.values():
                found = self._extract_tensor(nested)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for nested in value:
                found = self._extract_tensor(nested)
                if found is not None:
                    return found
        return None

    def _load_csv(self, asset: ActionAsset) -> list[list[float]]:
        with asset.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            first = next(reader, [])
            if first and first[0].lstrip().startswith("# joint_names"):
                header = next(reader, [])
                data_start = 2 if header[:2] == ["time_s", "phase"] else 0
                names: list[str] = []
            else:
                header = first
                data_start = 0
                names = [name for name in header if name]
            rows: list[list[float]] = []
            for row in reader:
                try:
                    rows.append([float(value) for value in row[data_start:]])
                except ValueError:
                    continue
        return self._map_array(rows, names)

    def _load_file(self, asset: ActionAsset) -> list[list[float]]:
        if asset.suffix == ".npz":
            if np is None:
                raise RuntimeError("numpy is not installed in the active conda environment")
            with np.load(asset.path, allow_pickle=True) as archive:
                if "qpos" not in archive:
                    raise ValueError("npz does not contain a qpos array")
                names = [str(v) for v in np.asarray(archive.get("joint_names", [])).reshape(-1)]
                return self._map_array(archive["qpos"], names)
        if asset.suffix == ".csv":
            return self._load_csv(asset)
        if asset.suffix == ".pt":
            if torch is None:
                raise RuntimeError("torch is required to decode .pt assets; install it in allrobotrl-platform")
            value = torch.load(asset.path, map_location="cpu", weights_only=False)
            tensor = self._extract_tensor(value)
            if tensor is None:
                raise ValueError(".pt is a policy/checkpoint and does not contain a qpos sequence")
            return self._map_array(tensor)
        if asset.suffix == ".pkl":
            with asset.path.open("rb") as handle:
                value = pickle.load(handle)
            tensor = self._extract_tensor(value)
            if tensor is None:
                raise ValueError("pkl does not contain a robot qpos sequence; run GMR retargeting first")
            return self._map_array(tensor)
        raise ValueError(f"unsupported asset type: {asset.suffix}")

    def frames(self, asset_id: str) -> list[list[float]]:
        with self._lock:
            asset = self.get(asset_id)
            if asset._frames is None:
                try:
                    asset._frames = self._load_file(asset)
                    asset.frame_count = len(asset._frames)
                    asset.nq = self.model.nq
                except Exception as exc:
                    asset.load_error = str(exc)
                    raise
            return asset._frames

    def frame(self, asset_id: str, frame: int) -> list[float]:
        values = self.frames(asset_id)
        index = max(0, min(len(values) - 1, int(frame)))
        return values[index]


class MuJoCoSession:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.overrides: dict[tuple[str, int], dict[str, float]] = {}
        self.keyframes: dict[str, dict] = {}
        self.joints = self._read_joints()
        self.joint_by_name = {joint["name"]: joint for joint in self.joints}
        self.actions = ActionLibrary(ACTION_ROOT, self.model, self.joints)

    def _read_joints(self) -> list[dict]:
        joints = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_type = int(self.model.jnt_type[joint_id])
            if not name or joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            limited = bool(self.model.jnt_limited[joint_id])
            lower = float(self.model.jnt_range[joint_id][0]) if limited else -math.pi
            upper = float(self.model.jnt_range[joint_id][1]) if limited else math.pi
            joints.append({"id": joint_id, "name": name, "type": "hinge" if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE) else "slide", "qposAdr": qpos_adr, "limited": limited, "lowerRad": lower, "upperRad": upper, "lowerDeg": math.degrees(lower), "upperDeg": math.degrees(upper), "defaultRad": float(self.model.qpos0[qpos_adr]), "defaultDeg": math.degrees(float(self.model.qpos0[qpos_adr]))})
        return joints

    def _base_pose(self) -> list[float]:
        pose = [float(value) for value in self.model.qpos0]
        if len(pose) >= 7:
            pose[0:7] = [0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0]
        return pose

    def _demo_pose(self, frame: int) -> list[float]:
        pose = self._base_pose()
        phase = 2.0 * math.pi * (frame % FRAME_COUNT) / FRAME_COUNT
        for joint in self.joints:
            name = joint["name"]
            value = joint["defaultRad"]
            if "hip_pitch" in name:
                side = -1.0 if name.startswith("left") else 1.0
                value = 0.18 * math.sin(phase + side * math.pi)
            elif "knee" in name:
                value = 0.34 + 0.22 * max(0.0, math.sin(phase + (math.pi if name.startswith("left") else 0.0)))
            elif "ankle_pitch" in name:
                value = -0.14 * math.sin(phase + (math.pi if name.startswith("left") else 0.0))
            elif "shoulder_pitch" in name:
                side = -1.0 if name.startswith("left") else 1.0
                value = 0.12 * math.sin(phase + side * math.pi)
            elif "elbow" in name:
                value = 0.8
            if joint["limited"]:
                value = clamp(value, joint["lowerRad"], joint["upperRad"])
            pose[joint["qposAdr"]] = value
        return pose

    def pose_for_frame(self, frame: int, asset_id: str | None = None) -> list[float]:
        if asset_id:
            pose = list(self.actions.frame(asset_id, frame))
            asset = self.actions.get(asset_id)
            frame = max(0, min(max(asset.frame_count - 1, 0), int(frame)))
        else:
            frame = max(0, min(FRAME_COUNT - 1, int(frame)))
            pose = self._demo_pose(frame)
        overrides = self.overrides.get((asset_id or "demo", frame), {})
        for name, value in overrides.items():
            joint = self.joint_by_name.get(name)
            if joint:
                pose[joint["qposAdr"]] = clamp(value, joint["lowerRad"], joint["upperRad"])
        return pose

    def frame_payload(self, frame: int, asset_id: str | None = None) -> dict:
        with self.lock:
            pose = self.pose_for_frame(frame, asset_id)
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            actual_frame = int(frame)
            if asset_id:
                actual_frame = max(0, min(self.actions.get(asset_id).frame_count - 1, actual_frame))
            joints = []
            for joint in self.joints:
                value = float(self.data.qpos[joint["qposAdr"]])
                joints.append({"name": joint["name"], "angleRad": value, "angleDeg": math.degrees(value), "lowerDeg": joint["lowerDeg"], "upperDeg": joint["upperDeg"], "limited": joint["limited"]})
            pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            if pelvis < 0:
                pelvis = 1
            return {"frame": actual_frame, "timeSec": actual_frame / FPS, "fps": FPS, "qpos": [float(value) for value in self.data.qpos], "joints": joints, "root": {"x": float(self.data.xpos[pelvis][0]), "y": float(self.data.xpos[pelvis][1]), "z": float(self.data.xpos[pelvis][2])}, "overrideCount": len(self.overrides.get((asset_id or "demo", actual_frame), {})), "engine": "MuJoCo", "assetId": asset_id, "modelPath": str(self.model_path)}

    def apply_joint(self, frame: int, name: str, angle_deg: float, asset_id: str | None = None) -> dict:
        with self.lock:
            joint = self.joint_by_name.get(name)
            if not joint:
                raise ValueError(f"unknown joint: {name}")
            if asset_id:
                asset = self.actions.get(asset_id)
                frame = max(0, min(asset.frame_count - 1, int(frame)))
            else:
                frame = max(0, min(FRAME_COUNT - 1, int(frame)))
            angle_rad = clamp(math.radians(float(angle_deg)), joint["lowerRad"], joint["upperRad"])
            self.overrides.setdefault((asset_id or "demo", frame), {})[name] = angle_rad
            payload = self.frame_payload(frame, asset_id)
            payload["editedJoint"] = {"name": name, "angleDeg": math.degrees(angle_rad), "angleRad": angle_rad}
            return payload

    def add_keyframe(self, frame: int, label: str | None = None, asset_id: str | None = None) -> dict:
        with self.lock:
            payload = self.frame_payload(frame, asset_id)
            prefix = asset_id[:6] if asset_id else "demo"
            keyframe_id = f"kf-{prefix}-{int(payload['frame']):04d}"
            keyframe = {"id": keyframe_id, "frame": int(payload["frame"]), "assetId": asset_id, "label": label or f"关键帧 {int(payload['frame']):03d}", "createdAt": iso_now(), "qpos": payload["qpos"], "jointCount": len(payload["joints"]), "engine": "MuJoCo"}
            self.keyframes[keyframe_id] = keyframe
            return keyframe

    def export_payload(self, body: dict) -> dict:
        asset_id = body.get("assetId")
        frames = [int(frame) for frame in body.get("frames", sorted({k["frame"] for k in self.keyframes.values()}))]
        if not frames:
            frames = [int(body.get("frame", 0))]
        return {"format": "mujoco.pose.v1", "engine": f"MuJoCo {RUNTIME_VERSION}", "model": {"name": "g1_mocap_29dof", "path": str(self.model_path), "urdfPath": str(URDF_PATH) if URDF_PATH else None, "nq": self.model.nq, "nv": self.model.nv, "jointCount": len(self.joints)}, "asset": self.actions.get(asset_id).public() if asset_id else None, "fps": FPS, "frames": [self.frame_payload(frame, asset_id) for frame in frames], "keyframes": [value for value in self.keyframes.values() if value.get("assetId") == asset_id], "metadata": body.get("metadata", {}), "exportedAt": iso_now()}


def encode_png(rgb: Any) -> bytes:
    """Encode an RGB numpy image without adding a Pillow runtime dependency."""
    if np is None:
        raise RuntimeError("numpy is required for MuJoCo rendering")
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("MuJoCo renderer returned an unexpected image shape")
    height, width, _ = image.shape
    rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    import zlib

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, level=6)) + chunk(b"IEND", b"")


class MuJoCoRenderer:
    """Off-screen renderer backed by the same MjModel used for physics/API."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.states_lock = threading.RLock()
        self.states: dict[int, dict[str, Any]] = {}
        self.closed = False
        # Fail fast at startup when EGL/OSMesa/GLFW is unavailable instead of
        # accepting requests that can never produce an image. Render calls on
        # Requests execute on this service's single HTTP worker; if the server
        # is embedded in another threaded host, each worker still gets a safe
        # context through _state().
        self._state()

    def _state(self) -> dict[str, Any]:
        thread_id = threading.get_ident()
        with self.states_lock:
            state = self.states.get(thread_id)
            if state is None:
                state = {
                    "data": mujoco.MjData(self.model),
                    "camera": mujoco.MjvCamera(),
                    "renderer": None,
                    "size": None,
                }
                state["camera"].type = mujoco.mjtCamera.mjCAMERA_FREE
                # Construct a small framebuffer now so startup validates the
                # selected backend. It is resized lazily for actual requests.
                state["renderer"] = mujoco.Renderer(self.model, height=640, width=640)
                state["size"] = (640, 640)
                self.states[thread_id] = state
            return state

    def _ensure_renderer(self, state: dict[str, Any], width: int, height: int) -> mujoco.Renderer:
        if state["renderer"] is None or state["size"] != (width, height):
            if state["renderer"] is not None:
                state["renderer"].close()
            state["renderer"] = mujoco.Renderer(self.model, height=height, width=width)
            state["size"] = (width, height)
        return state["renderer"]

    def render(self, pose: list[float], width: int = 640, height: int = 640, azimuth: float = 135.0, elevation: float = -25.0, distance: float = 2.35) -> bytes:
        width = max(240, min(1400, int(width)))
        height = max(240, min(1000, int(height)))
        if self.closed:
            raise RuntimeError("MuJoCo renderer is closed")
        state = self._state()
        data = state["data"]
        camera = state["camera"]
        data.qpos[:] = pose
        mujoco.mj_forward(self.model, data)
        renderer = self._ensure_renderer(state, width, height)
        camera.lookat[:] = [0.0, 0.0, 0.82]
        camera.distance = max(1.5, min(8.0, float(distance)))
        camera.azimuth = float(azimuth)
        camera.elevation = max(-89.0, min(89.0, float(elevation)))
        renderer.update_scene(data, camera=camera)
        return encode_png(renderer.render())

    def close(self) -> None:
        with self.states_lock:
            for state in self.states.values():
                if state["renderer"] is not None:
                    state["renderer"].close()
                    state["renderer"] = None
            self.states.clear()
            self.closed = True

    def reset(self) -> None:
        """Drop all renderer state before a new motion asset is displayed."""
        self.close()
        self.closed = False
        self._state()

    @property
    def render_ready(self) -> bool:
        with self.states_lock:
            return bool(self.states)


SESSION = MuJoCoSession(MODEL_PATH)
RENDERER = MuJoCoRenderer(SESSION.model)



def urdf_payload() -> dict:
    joints: list[dict] = []
    runtime_check: dict[str, Any] = {"loadedByMuJoCo": False}
    if URDF_PATH and URDF_PATH.exists():
        try:
            urdf_model = mujoco.MjModel.from_xml_path(str(URDF_PATH))
            runtime_check = {
                "loadedByMuJoCo": True,
                "nq": int(urdf_model.nq),
                "nv": int(urdf_model.nv),
                "njnt": int(urdf_model.njnt),
            }
            root = ET.parse(URDF_PATH).getroot()
            for node in root.findall("joint"):
                joint_type = node.attrib.get("type", "fixed")
                if joint_type == "fixed":
                    continue
                limit = node.find("limit")
                lower = float(limit.attrib.get("lower", "-3.141592653589793")) if limit is not None else -math.pi
                upper = float(limit.attrib.get("upper", "3.141592653589793")) if limit is not None else math.pi
                joints.append({"name": node.attrib.get("name", ""), "type": joint_type, "lowerDeg": math.degrees(lower), "upperDeg": math.degrees(upper)})
        except Exception as exc:
            runtime_check = {"loadedByMuJoCo": False, "loadError": str(exc)}
            if not joints:
                return {"path": str(URDF_PATH), "error": str(exc), "runtime": runtime_check, "joints": []}
    return {"path": str(URDF_PATH) if URDF_PATH else None, "joints": joints, "jointCount": len(joints), "runtime": runtime_check, "source": "third_party/GMR-master"}


class Handler(BaseHTTPRequestHandler):
    server_version = "MotionLabMuJoCo/0.2"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[mujoco] {self.address_string()} - {fmt % args}")

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=jsonable).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, status: int, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_html(200, f"<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Motion Lab MuJoCo Service</title><style>body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:820px;margin:48px auto;padding:0 24px;color:#20242b;line-height:1.6}}h1{{margin-bottom:8px}}.ok{{color:#16834b;font-weight:600}}code{{background:#f1f3f5;padding:2px 6px;border-radius:4px}}a{{color:#1261a0}}</style></head><body><h1>Motion Lab MuJoCo Service</h1><p class=\"ok\">服务运行正常</p><p>运行时：<strong>MuJoCo {RUNTIME_VERSION}</strong><br>仓库源码：<strong>third_party/mujoco-main · {MUJOCO_SOURCE_VERSION}</strong><br>模型：<code>{SESSION.model_path}</code><br>动作资产：<strong>{len(SESSION.actions.assets)}</strong></p><p><a href=\"/api/mujoco\">查看 API 路由</a> · <a href=\"/api/mujoco/actions\">查看动作资产</a> · <a href=\"http://127.0.0.1:4173/\">打开 React 工作台</a></p></body></html>")
            elif parsed.path == "/api/mujoco":
                self._send(200, {"service": "Motion Lab MuJoCo", "status": "ok", "runtime": RUNTIME_VERSION, "sourceVersion": MUJOCO_SOURCE_VERSION, "renderer": "MuJoCo Renderer", "frontend": "http://127.0.0.1:4173/", "routes": {"health": "/api/mujoco/health", "model": "/api/mujoco/model", "urdf": "/api/mujoco/urdf", "render": "/api/mujoco/render?asset={id}&frame={frame}", "reset": "/api/mujoco/session/reset", "actions": "/api/mujoco/actions", "actionFrame": "/api/mujoco/actions/{id}/frames/{frame}", "legacyFrames": "/api/mujoco/frames/{frame}", "keyframes": "/api/mujoco/keyframes", "export": "/api/mujoco/export"}})
            elif parts == ["api", "mujoco", "health"]:
                self._send(200, {"status": "ok", "engine": "MuJoCo", "runtimeVersion": RUNTIME_VERSION, "sourceVersion": MUJOCO_SOURCE_VERSION, "model": str(SOURCE_MODEL_PATH), "runtimeModel": str(SESSION.model_path), "urdf": str(URDF_PATH) if URDF_PATH else None, "renderer": "MuJoCo Renderer", "renderBackend": RENDER_BACKEND, "headless": RENDER_BACKEND in {"egl", "osmesa"}, "renderReady": RENDERER.render_ready, "python": sys.executable, "condaPrefix": os.environ.get("CONDA_PREFIX") or str(Path(sys.executable).resolve().parent), "timestamp": iso_now()})
            elif parts == ["api", "mujoco", "model"]:
                self._send(200, {"name": "g1_mocap_29dof", "engine": "MuJoCo", "runtimeVersion": RUNTIME_VERSION, "sourceVersion": MUJOCO_SOURCE_VERSION, "nq": SESSION.model.nq, "nv": SESSION.model.nv, "jointCount": len(SESSION.joints), "modelPath": str(SOURCE_MODEL_PATH), "runtimeModelPath": str(SESSION.model_path), "urdfPath": str(URDF_PATH) if URDF_PATH else None, "geomCount": SESSION.model.ngeom, "joints": SESSION.joints})
            elif parts == ["api", "mujoco", "urdf"]:
                self._send(200, urdf_payload())
            elif parts == ["api", "mujoco", "render"]:
                asset_id = query.get("asset", [None])[0]
                frame = int(query.get("frame", [0])[0])
                pose = SESSION.pose_for_frame(frame, asset_id)
                image = RENDERER.render(
                    pose,
                    width=int(query.get("width", [720])[0]),
                    height=int(query.get("height", [720])[0]),
                    azimuth=float(query.get("azimuth", [135])[0]),
                    elevation=float(query.get("elevation", [-25])[0]),
                    distance=float(query.get("distance", [2.35])[0]),
                )
                self._send_bytes(200, image, "image/png")
            elif parts == ["api", "mujoco", "actions"]:
                include_source = query.get("includeSource", ["0"])[0] in {"1", "true"}
                self._send(200, {"root": str(SESSION.actions.root), "count": len(SESSION.actions.assets), "assets": SESSION.actions.list_public(include_source)})
            elif len(parts) == 4 and parts[:3] == ["api", "mujoco", "actions"]:
                self._send(200, SESSION.actions.get(parts[3]).public())
            elif len(parts) == 5 and parts[:3] == ["api", "mujoco", "actions"] and parts[4] == "frames":
                asset_id = parts[3]
                start = max(0, int(query.get("start", [0])[0]))
                count = max(1, min(120, int(query.get("count", [30])[0])))
                asset = SESSION.actions.get(asset_id)
                self._send(200, {"asset": asset.public(), "frames": [SESSION.frame_payload(index, asset_id) for index in range(start, min(start + count, asset.frame_count))]})
            elif len(parts) == 6 and parts[:3] == ["api", "mujoco", "actions"] and parts[4] == "frames":
                self._send(200, SESSION.frame_payload(int(parts[5]), parts[3]))
            elif parts == ["api", "mujoco", "frames"]:
                count = max(1, min(FRAME_COUNT, int(query.get("count", [FRAME_COUNT])[0])))
                asset_id = query.get("asset", [None])[0]
                if asset_id:
                    asset = SESSION.actions.get(asset_id)
                    count = min(count, asset.frame_count)
                self._send(200, {"fps": FPS, "count": count, "assetId": asset_id, "frames": [{"frame": frame, "timeSec": frame / FPS, "keyframe": any(k["assetId"] == asset_id and k["frame"] == frame for k in SESSION.keyframes.values())} for frame in range(count)]})
            elif len(parts) == 4 and parts[:3] == ["api", "mujoco", "frames"]:
                self._send(200, SESSION.frame_payload(int(parts[3]), query.get("asset", [None])[0]))
            elif parts == ["api", "mujoco", "keyframes"]:
                self._send(200, {"keyframes": list(SESSION.keyframes.values())})
            else:
                self._send(404, {"error": "route_not_found"})
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def _safe_action_path(self, value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_relative_to(SESSION.actions.root):
            raise ValueError("action path must be inside MOTIONLAB_ACTION_ROOT")
        if not path.exists() or not path.is_file():
            raise ValueError("action file does not exist")
        if path.suffix.lower() not in ActionLibrary.EXTENSIONS:
            raise ValueError("unsupported action file extension")
        return path

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        try:
            body = self._read_json()
            if parts == ["api", "mujoco", "session", "reset"]:
                RENDERER.reset()
                self._send(200, {"status": "reset", "engine": "MuJoCo", "assetId": body.get("assetId"), "renderReady": RENDERER.render_ready})
            elif parts == ["api", "mujoco", "actions", "import"]:
                asset = SESSION.actions.register(self._safe_action_path(str(body["path"])))
                self._send(201, asset.public())
            elif len(parts) == 7 and parts[:3] == ["api", "mujoco", "actions"] and parts[4] == "frames" and parts[6] == "joints":
                self._send(200, SESSION.apply_joint(int(parts[5]), str(body["joint"]), float(body["angleDeg"]), parts[3]))
            elif len(parts) == 5 and parts[:3] == ["api", "mujoco", "actions"] and parts[4] == "keyframes":
                self._send(201, SESSION.add_keyframe(int(body.get("frame", 0)), body.get("label"), parts[3]))
            elif len(parts) == 5 and parts[:3] == ["api", "mujoco", "frames"] and parts[3] == "batch" and parts[4] == "joints":
                asset_id = body.get("assetId")
                frames = [int(frame) for frame in body.get("frames", [])]
                results = [SESSION.apply_joint(frame, str(body["joint"]), float(body["angleDeg"]), asset_id) for frame in frames]
                self._send(200, {"joint": str(body["joint"]), "assetId": asset_id, "frames": frames, "appliedCount": len(results), "current": results[-1] if results else None})
            elif len(parts) == 5 and parts[:3] == ["api", "mujoco", "frames"] and parts[4] == "joints":
                self._send(200, SESSION.apply_joint(int(parts[3]), str(body["joint"]), float(body["angleDeg"]), body.get("assetId")))
            elif parts == ["api", "mujoco", "keyframes"]:
                self._send(201, SESSION.add_keyframe(int(body.get("frame", 0)), body.get("label"), body.get("assetId")))
            elif parts == ["api", "mujoco", "export"]:
                self._send(200, SESSION.export_payload(body))
            else:
                self._send(404, {"error": "route_not_found"})
        except Exception as exc:
            self._send(400, {"error": str(exc)})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        try:
            if len(parts) == 7 and parts[:3] == ["api", "mujoco", "actions"] and parts[4] == "frames" and parts[6] == "joints":
                asset_id = parts[3]
                frame = int(parts[5])
                SESSION.overrides.pop((asset_id, frame), None)
                self._send(200, SESSION.frame_payload(frame, asset_id))
            elif len(parts) == 5 and parts[:3] == ["api", "mujoco", "frames"] and parts[4] == "joints":
                frame = int(parts[3])
                SESSION.overrides.pop(("demo", frame), None)
                self._send(200, SESSION.frame_payload(frame))
            elif len(parts) == 4 and parts[:3] == ["api", "mujoco", "keyframes"]:
                removed = SESSION.keyframes.pop(parts[3], None)
                self._send(200, {"removed": bool(removed), "keyframes": list(SESSION.keyframes.values())})
            else:
                self._send(404, {"error": "route_not_found"})
        except Exception as exc:
            self._send(400, {"error": str(exc)})


def main() -> None:
    port = int(os.environ.get("MOTIONLAB_MUJOCO_PORT", "8787"))
    host = os.environ.get("MOTIONLAB_MUJOCO_HOST", "127.0.0.1")
    # MuJoCo's OpenGL context is thread-affine. A single HTTP worker keeps the
    # context reusable and avoids WGL/EGL failures when the browser scrubs the
    # timeline quickly; expensive training work remains outside this service.
    server = HTTPServer((host, port), Handler)
    print(f"Motion Lab MuJoCo service listening on http://{host}:{port}")
    print(f"Conda prefix: {os.environ.get('CONDA_PREFIX', 'not-set')}")
    print(f"Source model: {SOURCE_MODEL_PATH}")
    print(f"Loaded MuJoCo runtime model: {SESSION.model_path}")
    print(f"Loaded URDF: {URDF_PATH or 'not found'}")
    print("Renderer: MuJoCo off-screen Renderer")
    print(f"Discovered action assets: {len(SESSION.actions.assets)}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        RENDERER.close()


if __name__ == "__main__":
    main()
