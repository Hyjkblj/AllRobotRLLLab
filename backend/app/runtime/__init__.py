"""External runtime adapters.

The platform process deliberately does not import Isaac Sim, GVHMR or GMR.
Those packages have incompatible Python/CUDA requirements and are executed
through the small, manifest-aware adapters in this package.
"""

from .contracts import (
    ExternalRunResult,
    RuntimeCheck,
    RuntimeUnavailable,
    RunnerError,
    Sim2SimExecution,
    TrainingExecution,
)
from .registry import RuntimeRegistry
from .gmr_runner import GmrRunner
from .gvhmr_runner import GVHMRRunner, GvhMrRunner
from .isaac_runner import IsaacLabRunner
from .mujoco_kinematics import MuJoCoKinematicsCompiler
from .unitree_sim2sim_runner import UnitreeMuJoCoRunner

__all__ = [
    "ExternalRunResult",
    "RuntimeCheck",
    "RuntimeRegistry",
    "RuntimeUnavailable",
    "RunnerError",
    "Sim2SimExecution",
    "TrainingExecution",
    "GmrRunner",
    "GVHMRRunner",
    "GvhMrRunner",
    "IsaacLabRunner",
    "MuJoCoKinematicsCompiler",
    "UnitreeMuJoCoRunner",
]
