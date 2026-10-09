"""Upgrade a legacy six-array motion archive to TrainMotionNPZ v1.

The legacy archive must already contain compiled world-space kinematics. This
tool only adds immutable identity metadata; it never fabricates or recomputes
robot motion. Joint and body ordering therefore requires an explicit operator
acknowledgement before the selected RobotSpec names are attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.application.robot_loader import load_robot_adapters  # noqa: E402
from backend.app.application.train_motion_validator import (  # noqa: E402
    TrainMotionFileError,
    validate_train_motion_file,
)
from backend.app.config.settings import Settings  # noqa: E402
from backend.app.domain.contracts import SchemaVersion  # noqa: E402


REQUIRED_ARRAYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
UPGRADER_VERSION = "legacy-train-motion-metadata-upgrade.v1"
BODY_ARRAYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_body_names(path: Path) -> tuple[str, ...]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainMotionFileError(f"unable to read source body names JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload or not all(isinstance(value, str) for value in payload):
        raise TrainMotionFileError("source body names JSON must be a non-empty string array")
    names = tuple(value.strip() for value in payload)
    if any(not name for name in names):
        raise TrainMotionFileError("source body names must not contain empty values")
    if len(set(names)) != len(names):
        raise TrainMotionFileError("source body names must be unique")
    return names


def _select_robot_bodies(
    arrays: dict[str, np.ndarray],
    *,
    robot_body_names: Sequence[str],
    source_body_names: Sequence[str] | None,
) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
    body_count = arrays["body_pos_w"].shape[1] if arrays["body_pos_w"].ndim == 3 else 0
    target_names = tuple(robot_body_names)
    if source_body_names is None:
        if body_count != len(target_names):
            raise TrainMotionFileError(
                f"legacy archive contains {body_count} bodies but RobotSpec requires {len(target_names)}; "
                "provide --source-body-names-file from the asset that produced the archive"
            )
        return arrays, tuple(range(body_count))

    source_names = tuple(source_body_names)
    if len(source_names) != body_count:
        raise TrainMotionFileError(
            f"source body names count {len(source_names)} does not match legacy body count {body_count}"
        )
    if len(set(source_names)) != len(source_names):
        raise TrainMotionFileError("source body names must be unique")
    missing = [name for name in target_names if name not in source_names]
    if missing:
        raise TrainMotionFileError(f"source body names do not contain RobotSpec bodies: {missing}")
    indexes = tuple(source_names.index(name) for name in target_names)
    selected = dict(arrays)
    for name in BODY_ARRAYS:
        value = arrays[name]
        if value.ndim < 2 or value.shape[1] != body_count:
            raise TrainMotionFileError(
                f"{name} body dimension does not match body_pos_w ({value.shape} versus {body_count})"
            )
        selected[name] = value[:, indexes, ...]
    return selected, indexes


def upgrade_legacy_train_motion(
    source: Path,
    output: Path,
    *,
    robot,
    assume_robot_spec_order: bool,
    source_body_names: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, object]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not assume_robot_spec_order:
        raise TrainMotionFileError(
            "legacy archives do not identify joint/body ordering; pass "
            "--assume-robot-spec-order only after verifying the producer contract"
        )
    if not source.is_file():
        raise TrainMotionFileError(f"legacy motion archive does not exist: {source}")
    if source == output:
        raise TrainMotionFileError("output must differ from the legacy source path")
    if output.exists() and not force:
        raise TrainMotionFileError(f"output already exists: {output}; pass --force to replace it")

    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = set(REQUIRED_ARRAYS).difference(archive.files)
            if "fps" not in archive.files:
                missing.add("fps")
            if missing:
                raise TrainMotionFileError(f"legacy archive is missing fields: {sorted(missing)}")
            # Copy the existing compiled arrays verbatim. This upgrader only
            # attaches identity metadata; it must not change motion values or
            # silently reduce their precision.
            arrays = {name: np.array(archive[name], copy=True) for name in REQUIRED_ARRAYS}
            fps_value = np.asarray(archive["fps"])
            if fps_value.size != 1:
                raise TrainMotionFileError("legacy fps must contain exactly one value")
            fps = float(fps_value.reshape(-1)[0])
    except TrainMotionFileError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise TrainMotionFileError(f"unable to read legacy motion archive: {exc}") from exc

    arrays, body_indexes = _select_robot_bodies(
        arrays,
        robot_body_names=robot.body_names,
        source_body_names=source_body_names,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    source_sha256 = sha256_file(source)
    payload = {
        **arrays,
        "format_version": np.asarray(SchemaVersion.TRAIN_MOTION.value),
        "robot_id": np.asarray(robot.robot_id),
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_names": np.asarray(robot.joint_names),
        "body_names": np.asarray(robot.body_names),
        "coord_frame": np.asarray("world_z_up"),
        "quat_convention": np.asarray("wxyz"),
        "source_motion_hash": np.asarray(source_sha256),
        "compiler_version": np.asarray(UPGRADER_VERSION),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        validate_train_motion_file(temporary, robot)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "source_sha256": source_sha256,
        "robot_id": robot.robot_id,
        "fps": fps,
        "frames": int(arrays["joint_pos"].shape[0]),
        "source_body_count": len(source_body_names) if source_body_names is not None else len(body_indexes),
        "selected_body_indexes": ",".join(str(index) for index in body_indexes),
        "compiler_version": UPGRADER_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--assume-robot-spec-order", action="store_true")
    parser.add_argument(
        "--source-body-names-file",
        type=Path,
        help="JSON string array in the exact body-column order of the legacy archive",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    registry = load_robot_adapters(
        repository_root=settings.repository_root,
        modules=settings.robot_adapter_modules,
    )
    robot = registry.get(args.robot_id).get_spec()
    try:
        result = upgrade_legacy_train_motion(
            args.source,
            args.output,
            robot=robot,
            assume_robot_spec_order=args.assume_robot_spec_order,
            source_body_names=(
                load_source_body_names(args.source_body_names_file) if args.source_body_names_file else None
            ),
            force=args.force,
        )
    except TrainMotionFileError as exc:
        parser.error(str(exc))
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "UPGRADER_VERSION",
    "load_source_body_names",
    "main",
    "sha256_file",
    "upgrade_legacy_train_motion",
]
