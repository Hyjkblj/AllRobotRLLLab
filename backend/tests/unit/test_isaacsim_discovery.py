from pathlib import Path

from scripts.check_external_runtime import discover_isaacsim_path


def test_isaacsim_python_package_is_discoverable() -> None:
    path = discover_isaacsim_path()
    # The platform environment used in CI may not include Isaac Sim.
    if path is not None:
        assert path.is_dir()
        assert path.name == "site-packages"
