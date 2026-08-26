from pathlib import Path

from adapters.unitree_g1_29dof import UnitreeG1Adapter


def test_g1_adapter_resolves_assets_from_external_gmr(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("GMR_PATH", str(root / "third_party" / "GMR-master"))
    adapter = UnitreeG1Adapter(repository_root=root)
    result = adapter.self_check()
    assert result.valid
    identity = adapter.asset_identity()
    assert identity["mujoco_xml_uri"]["exists"] is True
    assert identity["urdf_uri"]["exists"] is True
