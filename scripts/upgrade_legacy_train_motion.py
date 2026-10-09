"""Upgrade a legacy six-array motion archive to TrainMotionNPZ v1.

The legacy archive must already contain compiled world-space kinematics. This
tool only adds immutable identity metadata; it never fabricates or recomputes
robot motion. Joint and body ordering therefore requires an explicit operator
acknowledgement before the selected RobotSpec names are attached.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade_legacy_train_motion(
    source: Path,
    output: Path,
    *,
    robot,
    assume_robot_spec_order: bool,
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
        "compiler_version": UPGRADER_VERSION,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--assume-robot-spec-order", action="store_true")
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
            force=args.force,
        )
    except TrainMotionFileError as exc:
        parser.error(str(exc))
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["UPGRADER_VERSION", "main", "sha256_file", "upgrade_legacy_train_motion"]
