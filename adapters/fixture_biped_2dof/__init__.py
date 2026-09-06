"""Small synthetic adapter used for multi-robot contract tests.

This package is intentionally not part of the default deployment registry. It
proves that application services can execute a second RobotSpec without
special-casing a vendor model or importing a simulator runtime.
"""

from .fixture_adapter import FixtureBipedAdapter


def create_adapter(*, repository_root=None):
    return FixtureBipedAdapter(repository_root=repository_root)


__all__ = ["FixtureBipedAdapter", "create_adapter"]
