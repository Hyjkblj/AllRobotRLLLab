"""Validate the external runtime boundary before a GPU acceptance run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REQUIRED_PATHS = {
    "ISAACLAB_PATH": "Isaac Lab",
    "ISAACSIM_PATH": "Isaac Sim",
    "GMR_PATH": "GMR",
    "GVHMR_PATH": "GVHMR",
    "UNITREE_MUJOCO_PATH": "Unitree MuJoCo",
}

EXPECTED_REVISIONS = {
    "ISAACLAB_PATH": "3c6e67bb5",
    "GMR_PATH": "bb1bbe4",
    "GVHMR_PATH": "6ec3ca3",
    "UNITREE_MUJOCO_PATH": "ae6a840",
}


def git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "not-a-git-checkout"


def main() -> int:
    failures: list[str] = []
    for variable, label in REQUIRED_PATHS.items():
        raw = os.getenv(variable, "").strip()
        if not raw:
            failures.append(f"{variable} is not set ({label})")
            continue
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            failures.append(f"{variable} does not point to a directory: {path}")
            continue
        revision = git_revision(path)
        expected = EXPECTED_REVISIONS.get(variable)
        if expected and not (revision == expected or revision.startswith(expected)):
            failures.append(f"{variable} revision mismatch: expected {expected}, got {revision}")
        print(f"{label}: {path} [{revision}]")

    if failures:
        print("External runtime validation failed:", file=sys.stderr)
        print("\n".join(f" - {failure}" for failure in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
