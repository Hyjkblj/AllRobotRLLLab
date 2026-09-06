"""RobotSpec adapter for the public Unitree H1 MuJoCo model."""

from __future__ import annotations

from pathlib import Path

from adapters.common import JsonRobotAdapter


class UnitreeH1Adapter(JsonRobotAdapter):
    name = "unitree_h1_19dof"
    gmr_robot = "unitree_h1"
    sim2sim_command_env = "H1_SIM2SIM_COMMAND"
    asset_env = {"mujoco_xml_uri": "H1_MJCF_PATH", "urdf_uri": "H1_URDF_PATH"}
    asset_prefixes = ("third_party/mujoco_menagerie-main/",)

    def __init__(self, *, repository_root: Path | None = None, spec_path: Path | None = None) -> None:
        super().__init__(repository_root=repository_root, spec_path=spec_path or Path(__file__).with_name("robot_spec.json"))


__all__ = ["UnitreeH1Adapter"]
