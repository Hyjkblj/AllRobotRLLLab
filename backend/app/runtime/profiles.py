"""Shared runtime requirement profiles.

The platform has several process roles.  Keeping their runtime requirements in
one module prevents Settings, the CLI doctor and acceptance scripts from
drifting apart as new external runtimes are added.
"""

from __future__ import annotations

from typing import Final


RUNTIME_PROFILES: Final[dict[str, tuple[str, ...]]] = {
    # The API only orchestrates work and must stay importable without GPU SDKs.
    "api": (),
    # Direct trajectories use the platform MuJoCo compiler; no GMR/GVHMR
    # process is needed for this queue. The selected adapter's model asset is
    # checked separately.
    "motion-cpu": (),
    "motion-gpu": ("gmr", "gvhmr"),
    "isaac-gpu": ("isaac_lab", "isaac_sim", "unitree_rl_lab"),
    # Native platform task entrypoints intentionally do not require the
    # Unitree provider checkout.
    "native-isaac-gpu": ("isaac_lab", "isaac_sim"),
    "sim2sim-gpu": ("unitree_mujoco",),
    # The GPU worker consumes motion, Isaac and sim2sim queues in one image.
    "gpu": (
        "gmr",
        "gvhmr",
        "isaac_lab",
        "isaac_sim",
        "unitree_rl_lab",
        "unitree_mujoco",
    ),
}


def runtime_names(profile: str) -> tuple[str, ...]:
    """Return required runtime names for ``profile``.

    An invalid profile is a configuration error rather than an empty profile;
    silently accepting one would make a deployment appear ready without
    checking any external dependency.
    """

    try:
        return RUNTIME_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown runtime profile: {profile}") from exc


def infer_profile(*, platform_role: str, p3_backend: str, explicit: str | None = None) -> str:
    """Resolve a process runtime profile from deployment role and backend."""

    if explicit:
        runtime_names(explicit)
        return explicit
    if platform_role == "api":
        return "api"
    if platform_role == "worker-cpu":
        return "motion-cpu"
    if platform_role in {"gpu", "worker-gpu"}:
        return "gpu"
    if p3_backend == "unitree_mujoco":
        return "sim2sim-gpu"
    if p3_backend == "isaac_lab":
        return "native-isaac-gpu"
    if p3_backend == "unitree_rl_lab":
        return "isaac-gpu"
    if p3_backend == "gmr_gvhmr":
        return "motion-gpu"
    return "api"


__all__ = ["RUNTIME_PROFILES", "infer_profile", "runtime_names"]
