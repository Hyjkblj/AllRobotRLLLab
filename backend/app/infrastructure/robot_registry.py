"""User-provided robot asset registration for Local File Mode."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.domain.contracts import RobotSpec
from backend.app.infrastructure.local_file import FileLock


class RobotRegistryError(ValueError):
    pass


class LocalRobotRegistry:
    """Register immutable external asset packages without copying bytes."""

    def __init__(self, runtime_root: Path) -> None:
        self.root = Path(runtime_root).expanduser().resolve() / "robots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.lock = FileLock(self.root / "registry.lock")

    def add(self, package: Path, *, robot_id: str | None = None) -> dict[str, Any]:
        package = Path(package).expanduser().resolve()
        if not package.is_dir():
            raise RobotRegistryError(f"robot asset package is not a directory: {package}")
        files = [path for path in package.rglob("*") if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc"]
        if not files:
            raise RobotRegistryError(f"robot asset package is empty: {package}")
        if not any(path.suffix.lower() in {".urdf", ".xml", ".mjcf", ".usd", ".usda", ".usdc"} for path in files):
            raise RobotRegistryError("robot asset package must contain a URDF, MJCF/XML or USD model")
        identities: list[dict[str, Any]] = []
        for path in sorted(files):
            relative = path.relative_to(package).as_posix()
            if path.suffix.lower() in {".urdf", ".xml", ".mjcf"}:
                self._validate_xml(path, package)
            identities.append({"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
        spec_path = next((path for path in files if path.name == "robot_spec.json"), None)
        spec: dict[str, Any] | None = None
        if spec_path is not None:
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RobotRegistryError(f"invalid robot_spec.json: {spec_path}") from exc
            if not isinstance(spec, dict) or not spec.get("robot_id") or not spec.get("schema_version"):
                raise RobotRegistryError("robot_spec.json must contain robot_id and schema_version")
            try:
                spec = RobotSpec.model_validate(spec).model_dump(mode="json")
            except Exception as exc:
                raise RobotRegistryError("robot_spec.json does not satisfy RobotSpec") from exc
        effective_robot_id = robot_id or (str(spec["robot_id"]) if spec else package.name)
        canonical = json.dumps({"robot_id": effective_robot_id, "files": identities, "spec": spec}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        version_id = hashlib.sha256(canonical).hexdigest()
        record = {"schema_version": "robot_registry.v1", "robot_id": effective_robot_id, "version_id": version_id, "package_path": str(package), "registered_at": datetime.now(timezone.utc).isoformat(), "verification": "verified" if spec else "assets_only", "spec": spec, "files": identities}
        with self.lock:
            records = self._read()
            existing = next((item for item in records if item.get("version_id") == version_id), None)
            if existing is not None:
                return existing
            records.append(record)
            self._write(records)
        return record

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return self._read()

    @staticmethod
    def _validate_xml(path: Path, package: Path) -> None:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise RobotRegistryError(f"invalid XML asset: {path}") from exc
        compiler = root.find("./compiler")
        mesh_dir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
        texture_dir = compiler.attrib.get("texturedir", "") if compiler is not None else ""
        for element in root.iter():
            reference = element.attrib.get("filename") or element.attrib.get("file")
            if not reference or reference.startswith(("http://", "https://")):
                continue
            prefix = mesh_dir if element.tag == "mesh" else (texture_dir if element.tag == "texture" else "")
            target = (path.parent / prefix / reference).resolve()
            try:
                target.relative_to(package)
            except ValueError as exc:
                raise RobotRegistryError(f"asset reference escapes package: {path} -> {reference}") from exc
            if not target.exists():
                raise RobotRegistryError(f"missing asset reference: {path} -> {reference}")

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RobotRegistryError(f"invalid robot registry: {self.path}") from exc
        return value if isinstance(value, list) else []

    def _write(self, records: list[dict[str, Any]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=".registry.", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(records, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["LocalRobotRegistry", "RobotRegistryError"]
