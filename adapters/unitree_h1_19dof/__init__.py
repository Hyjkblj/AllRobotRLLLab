"""Unitree H1 19 DoF adapter."""

from .h1_adapter import UnitreeH1Adapter


def create_adapter(*, repository_root=None):
    return UnitreeH1Adapter(repository_root=repository_root)


__all__ = ["UnitreeH1Adapter", "create_adapter"]
