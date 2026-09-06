"""Unitree G1 29 DoF adapter."""

from .g1_adapter import UnitreeG1Adapter


def create_adapter(*, repository_root=None):
    return UnitreeG1Adapter(repository_root=repository_root)

__all__ = ["UnitreeG1Adapter", "create_adapter"]
