"""Framework-neutral in-memory motion array boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MotionArrays:
    """Trajectory arrays passed between adapters and application services.

    The fields intentionally use array-like values rather than importing a
    numerical runtime in the domain package.  NumPy validation belongs to the
    CPU application service; MuJoCo adapters can consume the same boundary.
    """

    fps: float
    joint_pos: object
    root_pos: object
    root_rot: object
    joint_names: tuple[str, ...]
    quat_convention: str = "xyzw"
    coord_frame: str = "world_z_up"

    def copy(self) -> "MotionArrays":
        return MotionArrays(
            fps=float(self.fps),
            joint_pos=self.joint_pos.copy(),
            root_pos=self.root_pos.copy(),
            root_rot=self.root_rot.copy(),
            joint_names=tuple(self.joint_names),
            quat_convention=self.quat_convention,
            coord_frame=self.coord_frame,
        )


__all__ = ["MotionArrays"]

