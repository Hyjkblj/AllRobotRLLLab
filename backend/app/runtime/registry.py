"""Registration and readiness checks for independent runtime environments."""

from __future__ import annotations

import json
import os
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.runtime.contracts import RuntimeCheck, RuntimeUnavailable


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    path_env: str
    revision: str | None = None
    python_env: str | None = None
    package_env: str | None = None
    required: bool = True


class RuntimeRegistry:
    """Resolve runtime registrations without importing their dependencies."""

    SPECS = (
        RuntimeSpec("gmr", "GMR_PATH", "bb1bbe4", "GMR_PYTHON"),
        RuntimeSpec("gvhmr", "GVHMR_PATH", "6ec3ca3", "GVHMR_PYTHON"),
        RuntimeSpec("isaac_lab", "ISAACLAB_PATH", "3c6e67bb5", "ISAAC_PYTHON"),
        RuntimeSpec("isaac_sim", "ISAACSIM_PATH", None, "ISAAC_PYTHON"),
        RuntimeSpec("unitree_rl_lab", "UNITREE_RL_LAB_PATH", None, "ISAAC_PYTHON", required=True),
        RuntimeSpec("unitree_mujoco", "UNITREE_MUJOCO_PATH", "ae6a840", "SIM2SIM_PYTHON"),
    )

    def __init__(self, *, manifest_path: Path | None = None, registration_path: Path | None = None) -> None:
        self.manifest_path = manifest_path
        self.registration_path = registration_path

    def _registrations(self) -> dict[str, dict[str, str]]:
        if self.registration_path is None or not self.registration_path.is_file():
            return {}
        try:
            payload = json.loads(self.registration_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def register(self, name: str, *, path: Path, python: str | None = None, revision: str | None = None) -> RuntimeCheck:
        spec = next((item for item in self.SPECS if item.name == name), None)
        if spec is None:
            raise RuntimeUnavailable(f"unknown runtime registration: {name}")
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeUnavailable(f"runtime path does not exist: {resolved}")
        if self.registration_path is None:
            raise RuntimeUnavailable("runtime registration storage is not configured")
        registrations = self._registrations()
        selected_revision = revision or self._git_revision(resolved) or f"content:{self._source_hash(resolved)}"
        registrations[name] = {"path": str(resolved), "python": str(python or ""), "revision": str(selected_revision)}
        self.registration_path.parent.mkdir(parents=True, exist_ok=True)
        self.registration_path.write_text(json.dumps(registrations, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return self.check(spec)

    @staticmethod
    def _git_revision(path: Path) -> str | None:
        root = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
        if root.returncode != 0:
            return None
        try:
            if Path(root.stdout.strip()).resolve() != path.resolve():
                return None
        except OSError:
            return None
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _source_hash(path: Path) -> str:
        digest = hashlib.sha256()
        for file in sorted(item for item in path.rglob("*") if item.is_file() and ".git" not in item.parts):
            digest.update(file.relative_to(path).as_posix().encode("utf-8"))
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()

    def check(self, spec: RuntimeSpec) -> RuntimeCheck:
        registration = self._registrations().get(spec.name, {})
        raw = os.getenv(spec.path_env, "").strip() or str(registration.get("path", "")).strip()
        if not raw:
            return RuntimeCheck(spec.name, False, False, expected_revision=spec.revision, errors=(f"{spec.path_env} is not configured",))
        path = Path(raw).expanduser().resolve()
        errors: list[str] = []
        expected_revision = str(registration.get("revision", "")).strip() or spec.revision
        if not path.is_dir():
            errors.append(f"{spec.path_env} does not point to a directory: {path}")
            return RuntimeCheck(spec.name, True, False, path=str(path), expected_revision=expected_revision, errors=tuple(errors))
        revision = self._git_revision(path)
        revision_match = bool(revision and expected_revision and (revision == expected_revision or revision.startswith(expected_revision)))
        if expected_revision and expected_revision.startswith("content:"):
            revision = f"content:{self._source_hash(path)}"
            revision_match = revision == expected_revision
        if expected_revision and not revision_match:
            errors.append(f"revision mismatch: expected {expected_revision}, got {revision or 'unknown'}")
        python = os.getenv(spec.python_env or "", "").strip() or str(registration.get("python", "")).strip() or None
        if python and not Path(python).expanduser().exists():
            errors.append(f"{spec.python_env} does not point to an executable: {python}")
        return RuntimeCheck(spec.name, True, not errors, path=str(path), revision=revision, expected_revision=expected_revision, python=python, errors=tuple(errors))

    def doctor(self, *, required_only: bool = False) -> dict[str, Any]:
        checks = [self.check(spec) for spec in self.SPECS if not required_only or spec.required]
        failures = [check.as_dict() for check in checks if not check.available]
        manifest = None
        if self.manifest_path and self.manifest_path.is_file():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                failures.append({"name": "runtime_manifest", "status": "INVALID", "errors": ["manifest is not valid JSON"]})
        return {"status": "READY" if not failures else "NOT_READY", "checks": [check.as_dict() for check in checks], "failures": failures, "manifest": manifest}

    def require(self, name: str) -> RuntimeCheck:
        spec = next((item for item in self.SPECS if item.name == name), None)
        if spec is None:
            raise RuntimeUnavailable(f"unknown runtime registration: {name}")
        check = self.check(spec)
        if not check.available:
            raise RuntimeUnavailable(f"runtime '{name}' is not ready", details=check.as_dict())
        return check


__all__ = ["RuntimeRegistry", "RuntimeSpec"]
