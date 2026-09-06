"""Validate the external runtime boundary before a GPU acceptance run."""

from __future__ import annotations

import os
import subprocess
import sys
import argparse
import json
import importlib.util
import hashlib
from pathlib import Path

# Support the documented ``python scripts/...`` entry point, where Python's
# import root is the scripts directory rather than the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.runtime.profiles import RUNTIME_PROFILES, infer_profile, runtime_names  # noqa: E402
from backend.app.runtime.registry import RuntimeRegistry  # noqa: E402


# Compatibility exports for operators/scripts that imported these names.  The
# source of truth is RuntimeRegistry.SPECS plus runtime profiles below.
REQUIRED_PATHS = {spec.path_env: spec.name for spec in RuntimeRegistry.SPECS}
EXPECTED_REVISIONS = {spec.path_env: spec.revision for spec in RuntimeRegistry.SPECS if spec.revision}
OPTIONAL_PATHS = {spec.path_env: spec.name for spec in RuntimeRegistry.SPECS if spec.name not in runtime_names("gpu")}


def discover_isaacsim_path() -> Path | None:
    """Resolve pip-installed Isaac Sim when no launcher directory is present."""
    spec = importlib.util.find_spec("isaacsim")
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin).resolve()
    # ``.../site-packages/isaacsim/__init__.py`` is the standard package
    # layout.  Keep the site-packages root as the runtime mount identity.
    return origin.parent.parent if origin.parent.name == "isaacsim" else origin.parent


def git_revision(path: Path) -> str:
    root = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True)
    if root.returncode != 0:
        return "not-a-git-checkout"
    try:
        if Path(root.stdout.strip()).resolve() != path.resolve():
            return "not-a-git-checkout"
    except OSError:
        return "not-a-git-checkout"
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-checkout"


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file() and ".git" not in item.parts):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--registration", type=Path, default=Path(".runtime/runtime-registrations.json"), help="runtime registration file created by robotlab runtime register")
    parser.add_argument("--profile", choices=tuple(RUNTIME_PROFILES), default=None, help="runtime requirement profile; defaults to RUNTIME_PROFILE or the process role/backend")
    args = parser.parse_args()
    explicit_profile = args.profile or os.getenv("RUNTIME_PROFILE", "").strip().lower() or None
    has_process_context = bool(os.getenv("PLATFORM_ROLE", "").strip() or os.getenv("P3_BACKEND", "").strip())
    profile = explicit_profile or ("gpu" if not has_process_context else infer_profile(
        platform_role=os.getenv("PLATFORM_ROLE", "api").strip().lower(),
        p3_backend=os.getenv("P3_BACKEND", "fake_smoke").strip().lower(),
    ))
    required_names = set(runtime_names(profile))
    registrations: dict = {}
    if args.registration.is_file():
        try:
            value = json.loads(args.registration.read_text(encoding="utf-8"))
            registrations = value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            registrations = {}
    failures: list[str] = []
    results: dict[str, dict[str, str | bool | None]] = {}
    for spec in RuntimeRegistry.SPECS:
        variable = spec.path_env
        label = spec.name
        registration = registrations.get(spec.name, {}) if isinstance(registrations.get(spec.name, {}), dict) else {}
        raw = os.getenv(variable, "").strip() or str(registration.get("path", "")).strip()
        if variable == "ISAACSIM_PATH" and not raw:
            discovered = discover_isaacsim_path()
            if discovered is not None:
                raw = str(discovered)
        if not raw:
            if spec.name in required_names:
                failures.append(f"{variable} is not set ({label}; profile={profile})")
            results[variable] = {"label": label, "configured": False, "path": None, "revision": None}
            continue
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            failures.append(f"{variable} does not point to a directory: {path}")
            results[variable] = {"label": label, "configured": True, "path": str(path), "revision": None}
            continue
        revision = git_revision(path)
        expected = str(registration.get("revision", "")).strip() or spec.revision
        if expected and expected.startswith("content:"):
            revision = "content:" + source_hash(path)
        if expected and not (revision == expected or revision.startswith(expected)):
            failures.append(f"{variable} revision mismatch: expected {expected}, got {revision}")
        results[variable] = {"label": label, "configured": True, "path": str(path), "revision": revision, "expected": expected}

    if args.json:
        print(json.dumps({"status": "ok" if not failures else "failed", "profile": profile, "checks": results, "failures": failures}, ensure_ascii=False, indent=2))
    else:
        for item in results.values():
            if item.get("configured"):
                print(f"{item['label']}: {item['path']} [{item['revision']}]")

    if failures:
        print("External runtime validation failed:", file=sys.stderr)
        print("\n".join(f" - {failure}" for failure in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
