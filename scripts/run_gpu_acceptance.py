"""Guarded entrypoint for the server-side GPU acceptance gate.

The production Isaac/RSL-RL and Unitree MuJoCo adapters are intentionally not
implemented in the CPU platform image. This script makes that boundary
explicit so a CI runner cannot report a false production pass.
"""

from __future__ import annotations

import os
import shutil
import sys


def main() -> int:
    run_id = os.getenv("ALLROBOTRL_RUN_ID", "").strip()
    if not run_id:
        print("ALLROBOTRL_RUN_ID is required", file=sys.stderr)
        return 2
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi is required on the GPU acceptance runner", file=sys.stderr)
        return 2
    print(f"GPU acceptance harness reserved for run {run_id}")
    print("Real Isaac Lab/RSL-RL and Unitree MuJoCo adapters must be wired before this gate can pass.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
