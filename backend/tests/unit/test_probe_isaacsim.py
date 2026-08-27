from scripts.probe_isaacsim import PHYSICS_EXTENSIONS


def test_isaacsim_probe_uses_physics_only_extension_set() -> None:
    assert "omni.physx" in PHYSICS_EXTENSIONS
    assert "omni.physx.tensors" in PHYSICS_EXTENSIONS
    assert "omni.hydra.rtx" not in PHYSICS_EXTENSIONS
