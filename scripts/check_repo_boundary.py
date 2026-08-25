"""Fail if generated files or third-party source are tracked by Git."""

from __future__ import annotations

import subprocess
import sys


ALLOWED_THIRD_PARTY = {"third_party/README.md"}
FORBIDDEN_MARKERS = (
    "/node_modules/",
    "/__pycache__/",
    "/.runtime/",
    "/dist/",
    "/build/",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [path for path in result.stdout.decode("utf-8").split("\0") if path]


def main() -> int:
    paths = tracked_files()
    violations = [
        path
        for path in paths
        if (path.startswith("third_party/") and path not in ALLOWED_THIRD_PARTY)
        or any(marker in f"/{path}" for marker in FORBIDDEN_MARKERS)
    ]
    if violations:
        print("Repository boundary violations:", file=sys.stderr)
        print("\n".join(f" - {path}" for path in violations), file=sys.stderr)
        return 1
    print(f"Repository boundary OK: {len(paths)} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
