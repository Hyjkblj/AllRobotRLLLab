"""Resolve a MuJoCo preview model from deployment configuration.

The preview service is intentionally a thin HTTP shell. Model identity and
asset paths come from an explicit RobotSpec or environment overrides, so the
service can be reused for another robot without changing Python code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MuJoCoModelConfig:
    robot_id: str
    model_path: Path
    urdf_path: Path | None
    model_name: str
    root_body: str
    root_height: float


def _resolve_uri(value: str, *, repository_root: Path) -> Path:
    external_prefixes = (
        ("third_party/GMR-master/", "GMR_PATH"),
        ("third_party/mujoco_menagerie-main/", "MUJOCO_MENAGERIE_PATH"),
    )
    for prefix, env_name in external_prefixes:
        if value.startswith(prefix):
            external = os.getenv(env_name, "").strip()
            if external:
                return (Path(external).expanduser() / value[len(prefix):]).resolve()
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else repository_root / candidate).resolve()


def _spec_assets(spec_path: Path, *, repository_root: Path) -> tuple[str | None, Path | None, Path | None, str | None, float | None]:
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unable to read MOTIONLAB_ROBOT_SPEC: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MOTIONLAB_ROBOT_SPEC must contain a JSON object")
    robot_id = str(payload.get("robot_id", "")).strip() or None
    assets = payload.get("assets") if isinstance(payload.get("assets"), dict) else {}
    model_uri = next((assets.get(key) for key in ("mujoco_xml_uri", "mujoco_xml") if isinstance(assets.get(key), str) and assets.get(key)), None)
    urdf_uri = next((assets.get(key) for key in ("urdf_uri", "urdf") if isinstance(assets.get(key), str) and assets.get(key)), None)
    body_names = payload.get("body_names") if isinstance(payload.get("body_names"), list) else []
    root_body = str(body_names[0]).strip() if body_names and str(body_names[0]).strip() else None
    try:
        root_height = float(payload["default_root_height"]) if payload.get("default_root_height") is not None else None
    except (TypeError, ValueError):
        root_height = None
    return robot_id, _resolve_uri(model_uri, repository_root=repository_root) if model_uri else None, _resolve_uri(urdf_uri, repository_root=repository_root) if urdf_uri else None, root_body, root_height


def resolve_mujoco_model_config(*, repository_root: Path) -> MuJoCoModelConfig:
    root = repository_root.resolve()
    robot_id = os.getenv("MOTIONLAB_ROBOT_ID", "").strip() or None
    spec_configured = bool(os.getenv("MOTIONLAB_ROBOT_SPEC", "").strip())
    model_path = Path(os.getenv("MOTIONLAB_MODEL_PATH", "").strip()).expanduser() if os.getenv("MOTIONLAB_MODEL_PATH", "").strip() else None
    urdf_path = Path(os.getenv("MOTIONLAB_URDF_PATH", "").strip()).expanduser() if os.getenv("MOTIONLAB_URDF_PATH", "").strip() else None
    spec_path_raw = os.getenv("MOTIONLAB_ROBOT_SPEC", "").strip()
    spec_root_body = None
    spec_root_height = None
    if spec_path_raw:
        spec_robot_id, spec_model, spec_urdf, spec_root_body, spec_root_height = _spec_assets(Path(spec_path_raw).expanduser().resolve(), repository_root=root)
        robot_id = robot_id or spec_robot_id
        model_path = model_path or spec_model
        urdf_path = urdf_path or spec_urdf
    if model_path is not None and not model_path.is_absolute():
        model_path = root / model_path
    if urdf_path is not None and not urdf_path.is_absolute():
        urdf_path = root / urdf_path
    if model_path is None:
        # The G1 fallback exists solely for the legacy editor prototype. Once
        # a different robot identity or RobotSpec is selected, silently
        # loading the G1 model would produce a misleading preview and could
        # corrupt downstream joint/body mappings.
        if spec_configured or (robot_id is not None and robot_id != "unitree_g1_29dof"):
            identity = robot_id or "the selected RobotSpec"
            raise FileNotFoundError(
                f"MuJoCo model is not configured for {identity}; set MOTIONLAB_MODEL_PATH or declare assets.mujoco_xml_uri in MOTIONLAB_ROBOT_SPEC"
            )
        robot_id = robot_id or "unitree_g1_29dof"
        candidates = (
            root / "third_party" / "GMR-master" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
            Path(os.getenv("MOTIONLAB_ACTION_ROOT", "")).expanduser() / "GMR" / "assets" / "unitree_g1" / "g1_mocap_29dof.xml",
        )
        model_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model was not found: {model_path}")
    if urdf_path is not None:
        urdf_path = urdf_path.resolve()
    return MuJoCoModelConfig(
        robot_id=robot_id or "robot",
        model_path=model_path,
        urdf_path=urdf_path if urdf_path and urdf_path.is_file() else None,
        model_name=os.getenv("MOTIONLAB_MODEL_NAME", "").strip() or model_path.stem,
        root_body=os.getenv("MOTIONLAB_ROOT_BODY", "").strip() or spec_root_body or "pelvis",
        root_height=float(os.getenv("MOTIONLAB_ROOT_HEIGHT", "")) if os.getenv("MOTIONLAB_ROOT_HEIGHT", "").strip() else (spec_root_height if spec_root_height is not None else 0.79),
    )


__all__ = ["MuJoCoModelConfig", "resolve_mujoco_model_config"]
