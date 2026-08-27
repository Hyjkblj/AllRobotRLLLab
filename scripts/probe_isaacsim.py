"""Probe a pip-installed Isaac Sim runtime without running a training task.

The default mode follows Isaac Sim's physics-only headless example: it uses
``experience=None`` and enables only the extensions needed for PhysX updates.
This avoids treating a renderer/RTX shutdown crash as a training failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PHYSICS_EXTENSIONS = (
    "omni.physx",
    "omni.physx.tensors",
    "omni.physx.fabric",
    "omni.warp.core",
    "usdrt.scenegraph",
    "omni.kit.telemetry",
    "omni.kit.loop",
    "omni.kit.usd.mdl",
    "omni.usd.metrics.assembler.ui",
)


def _discover_package_root() -> Path | None:
    import importlib.util

    spec = importlib.util.find_spec("isaacsim")
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin).resolve()
    return origin.parent.parent if origin.parent.name == "isaacsim" else origin.parent


def run(*, frames: int, close: bool, gpu: int) -> dict[str, object]:
    package_root = _discover_package_root()
    if package_root is None:
        raise RuntimeError("isaacsim Python package is not importable in the active environment")

    from isaacsim import SimulationApp

    extra_args: list[str] = []
    for extension in PHYSICS_EXTENSIONS:
        extra_args.extend(("--enable", extension))

    started = time.perf_counter()
    app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": gpu,
            "physics_gpu": gpu,
            "extra_args": extra_args,
        },
        experience=None,
    )
    startup_seconds = time.perf_counter() - started
    result: dict[str, object] = {
        "schema_version": "isaacsim_probe.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "gpu": gpu,
        "startup": "passed",
        "startup_seconds": round(startup_seconds, 3),
        "frames_requested": frames,
        "frames_updated": 0,
        "close": "not_requested",
    }
    print(json.dumps({"event": "startup", **result}, ensure_ascii=False), flush=True)

    for _ in range(frames):
        app.update()
    result["frames_updated"] = frames
    result["update"] = "passed"
    print(json.dumps({"event": "update", **result}, ensure_ascii=False), flush=True)

    if close:
        app.close()
        result["close"] = "passed"
        print(json.dumps({"event": "close", **result}, ensure_ascii=False), flush=True)
    else:
        result["close"] = "skipped"
        print(json.dumps({"event": "close", **result}, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--close", action="store_true", help="exercise graceful shutdown; may expose an upstream RTX crash")
    parser.add_argument("--output", type=Path, help="write the final JSON result to this path")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    try:
        result = run(frames=args.frames, close=args.close, gpu=args.gpu)
    except Exception as exc:
        print(json.dumps({"event": "error", "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # If close was requested and returned, the process can terminate normally.
    # Without --close, an explicit exit avoids interpreter finalizers touching
    # Kit objects after a successful physics-only probe.
    if not args.close:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
