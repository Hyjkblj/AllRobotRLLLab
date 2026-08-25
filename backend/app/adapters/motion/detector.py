"""Content-aware direct motion detectors.

Detectors inspect file content and array shapes instead of trusting the file
extension.  They return contracts and stable error codes; callers can decide
whether to persist the descriptor or show a user-facing validation result.
"""

from __future__ import annotations

import csv
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from backend.app.domain.contracts import (
    ArrayField,
    LicenseInfo,
    SourceMotionDescriptor,
)


class MotionDetectionError(ValueError):
    """A stable, user-actionable detection failure."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _license() -> LicenseInfo:
    return LicenseInfo(status="declared", source="user", processing_scope="platform motion processing")


def _array_field(path: str, value: Any, *, include_value: bool = False) -> ArrayField:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise MotionDetectionError("SCHEMA_INVALID", f"object dtype is not allowed for {path}")
    if not np.issubdtype(array.dtype, np.number):
        raise MotionDetectionError("SCHEMA_INVALID", f"numeric array required for {path}")
    if not np.isfinite(array).all():
        raise MotionDetectionError("NONFINITE_VALUE", f"NaN or Inf found in {path}")
    shape = list(array.shape) or [1]
    return ArrayField(path=path, shape=shape, dtype=str(array.dtype), value=array.reshape(-1)[0].item() if include_value and array.size else None)


class Detector(Protocol):
    name: str
    extensions: tuple[str, ...]

    def inspect(self, path: Path, *, asset_version_id: str, trusted_pickle: bool = False) -> SourceMotionDescriptor | None: ...


class NpzDetector:
    name = "npz-motion-detector.v1"
    extensions = (".npz",)

    def inspect(self, path: Path, *, asset_version_id: str, trusted_pickle: bool = False) -> SourceMotionDescriptor | None:
        try:
            archive = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise MotionDetectionError("SCHEMA_INVALID", f"unable to read NPZ: {exc}") from exc
        try:
            keys = set(archive.files)
            for name in archive.files:
                if archive[name].dtype.kind == "O":
                    raise MotionDetectionError("SCHEMA_INVALID", f"object dtype is not allowed in NPZ field: {name}")
            if "smpl_params_global" in keys:
                value = archive["smpl_params_global"]
                return SourceMotionDescriptor(
                    asset_version_id=asset_version_id,
                    file_format="npz",
                    detected_type="gvhmr_result",
                    source_skeleton="smpl",
                    fields={"smpl_params_global": _array_field("smpl_params_global", value)},
                    coord_frame="world_z_up",
                    quaternion_convention="xyzw",
                    license=_license(),
                    detector_version=self.name,
                )
            candidate_name = next((name for name in ("joint_pos", "dof_pos", "qpos") if name in keys), None)
            if candidate_name is not None:
                values = archive[candidate_name]
                if values.ndim != 2 or values.shape[0] < 15 or values.shape[1] not in (29, 36):
                    raise MotionDetectionError("SCHEMA_INVALID", "G1 trajectory must have shape [N, 29] or [N, 36]", details={"shape": list(values.shape)})
                names: list[str] = []
                if "joint_names" in keys:
                    raw_names = archive["joint_names"]
                    if raw_names.dtype.kind == "O":
                        raise MotionDetectionError("SCHEMA_INVALID", "joint_names object dtype is not allowed")
                    names = [str(item) for item in raw_names.reshape(-1)]
                fps = None
                if "fps" in keys:
                    fps = float(np.asarray(archive["fps"]).reshape(-1)[0])
                    if not 15 <= fps <= 120:
                        raise MotionDetectionError("SCHEMA_INVALID", "fps must be between 15 and 120", details={"fps": fps})
                fields = {candidate_name: _array_field(candidate_name, values)}
                if "fps" in keys:
                    fields["fps"] = _array_field("fps", archive["fps"], include_value=True)
                return SourceMotionDescriptor(
                    asset_version_id=asset_version_id,
                    file_format="npz",
                    detected_type="g1_joint_trajectory",
                    source_skeleton="unitree_g1_29dof",
                    fields=fields,
                    joint_names=names,
                    coord_frame="world_z_up",
                    quaternion_convention="xyzw",
                    license=_license(),
                    detector_version=self.name,
                )
            pose_key = next((name for name in ("joints", "joint_positions", "poses", "transl") if name in keys), None)
            if pose_key is not None:
                values = archive[pose_key]
                if values.ndim < 3 or values.shape[0] < 15:
                    raise MotionDetectionError("SCHEMA_INVALID", "human pose trajectory must contain [N, J, ...] frames", details={"shape": list(values.shape)})
                return SourceMotionDescriptor(
                    asset_version_id=asset_version_id,
                    file_format="npz",
                    detected_type="human_pose",
                    source_skeleton="human_pose",
                    fields={pose_key: _array_field(pose_key, values)},
                    coord_frame="world_z_up",
                    quaternion_convention="xyzw",
                    license=_license(),
                    detector_version=self.name,
                )
            return None
        finally:
            archive.close()


class CsvDetector:
    name = "csv-motion-detector.v1"
    extensions = (".csv",)

    def inspect(self, path: Path, *, asset_version_id: str, trusted_pickle: bool = False) -> SourceMotionDescriptor | None:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                first = next(reader, [])
                if not first:
                    raise MotionDetectionError("SCHEMA_INVALID", "CSV has no header")
                joint_names: list[str] = []
                if first[0].lstrip().lower().startswith("# joint_names"):
                    joint_names = [name.strip() for name in first[1:] if name.strip()]
                    header = next(reader, [])
                else:
                    header = first
                if header[:2] == ["time_s", "phase"]:
                    columns = header[2:]
                elif header and header[0] in {"time_s", "frame"}:
                    columns = header[1:]
                else:
                    columns = header
                if len(columns) != 29:
                    return None
                if joint_names and len(joint_names) != 29:
                    raise MotionDetectionError("SCHEMA_INVALID", "# joint_names must declare 29 names")
                if not joint_names:
                    joint_names = columns
                if len(set(joint_names)) != len(joint_names):
                    raise MotionDetectionError("SCHEMA_INVALID", "CSV joint names must be unique")
                frames = 0
                for row in reader:
                    if len(row) != len(header):
                        raise MotionDetectionError("SCHEMA_INVALID", "CSV row column count does not match header", details={"row": frames + 1})
                    values = [float(value) for value in (row[2:] if header[:2] == ["time_s", "phase"] else row[-29:])]
                    if not np.isfinite(values).all():
                        raise MotionDetectionError("NONFINITE_VALUE", "CSV contains NaN or Inf")
                    frames += 1
                if frames < 15:
                    raise MotionDetectionError("SCHEMA_INVALID", "trajectory must contain at least 15 frames", details={"frames": frames})
        except MotionDetectionError:
            raise
        except (OSError, ValueError, UnicodeError) as exc:
            raise MotionDetectionError("SCHEMA_INVALID", f"unable to read CSV: {exc}") from exc
        return SourceMotionDescriptor(
            asset_version_id=asset_version_id,
            file_format="csv",
            detected_type="g1_joint_trajectory",
            source_skeleton="unitree_g1_29dof",
            fields={"joint_pos": ArrayField(path="rows[-29:]", shape=[frames, 29], dtype="float64")},
            joint_names=joint_names,
            coord_frame="world_z_up",
            quaternion_convention="xyzw",
            license=_license(),
            detector_version=self.name,
        )


class PtDetector:
    name = "pt-motion-detector.v1"
    extensions = (".pt",)

    def inspect(self, path: Path, *, asset_version_id: str, trusted_pickle: bool = False) -> SourceMotionDescriptor | None:
        try:
            import torch
        except ImportError as exc:
            raise MotionDetectionError("PROCESSOR_UNAVAILABLE", "torch is required to inspect .pt files") from exc
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise MotionDetectionError("SCHEMA_INVALID", f"safe .pt loading failed: {exc}") from exc
        if hasattr(payload, "detach"):
            payload = {"joint_pos": payload}
        if not isinstance(payload, dict):
            raise MotionDetectionError("UNSUPPORTED_SOURCE_TYPE", "a .pt file must contain a tensor or tensor dictionary")
        for key in ("joint_pos", "dof_pos", "qpos"):
            if key in payload and hasattr(payload[key], "shape"):
                values = payload[key].detach().cpu().numpy()
                if values.ndim == 2 and values.shape[1] in (29, 36):
                    return SourceMotionDescriptor(
                        asset_version_id=asset_version_id,
                        file_format="pt",
                        detected_type="g1_joint_trajectory",
                        source_skeleton="unitree_g1_29dof",
                        fields={key: _array_field(key, values)},
                        coord_frame="world_z_up",
                        quaternion_convention="xyzw",
                        license=_license(),
                        detector_version=self.name,
                    )
        return None


class PklDetector:
    name = "pkl-motion-detector.v1"
    extensions = (".pkl",)

    def inspect(self, path: Path, *, asset_version_id: str, trusted_pickle: bool = False) -> SourceMotionDescriptor | None:
        if not trusted_pickle:
            raise MotionDetectionError("UNTRUSTED_PICKLE", "ordinary uploads cannot be parsed as pickle; use NPZ, CSV or PT")
        try:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise MotionDetectionError("SCHEMA_INVALID", f"trusted pickle loading failed: {exc}") from exc
        if isinstance(payload, dict):
            for key in ("joint_pos", "dof_pos", "qpos"):
                if key in payload:
                    values = np.asarray(payload[key])
                    if values.ndim == 2 and values.shape[1] in (29, 36):
                        return SourceMotionDescriptor(asset_version_id=asset_version_id, file_format="pkl", detected_type="g1_joint_trajectory", source_skeleton="unitree_g1_29dof", fields={key: _array_field(key, values)}, coord_frame="world_z_up", quaternion_convention="xyzw", license=_license(), detector_version=self.name)
        return None


@dataclass(frozen=True)
class MotionSourceRegistry:
    detectors: tuple[Detector, ...] = (NpzDetector(), CsvDetector(), PtDetector(), PklDetector())

    def detect(self, path: Path, *, asset_version_id: str | None = None, trusted_pickle: bool = False) -> SourceMotionDescriptor:
        resolved = path.resolve()
        if not resolved.is_file():
            raise MotionDetectionError("INPUT_NOT_FOUND", f"motion source does not exist: {resolved}")
        suffix = resolved.suffix.lower()
        matching = [detector for detector in self.detectors if suffix in detector.extensions]
        if not matching:
            raise MotionDetectionError("UNSUPPORTED_SOURCE_TYPE", f"unsupported motion extension: {suffix}")
        matches: list[SourceMotionDescriptor] = []
        for detector in matching:
            descriptor = detector.inspect(resolved, asset_version_id=asset_version_id or sha256_file(resolved), trusted_pickle=trusted_pickle)
            if descriptor is not None:
                matches.append(descriptor)
        if len(matches) == 0:
            raise MotionDetectionError("UNSUPPORTED_SOURCE_TYPE", "file content does not match a supported motion schema")
        if len(matches) > 1:
            raise MotionDetectionError("AMBIGUOUS_SOURCE_TYPE", "file matches more than one motion schema; choose a source type explicitly")
        return matches[0]


__all__ = ["MotionDetectionError", "MotionSourceRegistry", "sha256_file"]
