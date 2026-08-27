from scripts.collect_runtime_manifest import _asset_identity


def test_runtime_manifest_records_isaac_training_urdf(monkeypatch, tmp_path) -> None:
    urdf = tmp_path / "g1_29dof_rev_1_0.urdf"
    urdf.write_text("<robot name='g1'/>", encoding="utf-8")
    monkeypatch.setenv("G1_ISAAC_URDF_PATH", str(urdf))

    identity = _asset_identity(tmp_path)

    assert identity["isaac_urdf"]["path"] == str(urdf.resolve())
    assert identity["isaac_urdf"]["size_bytes"] == urdf.stat().st_size
    assert len(identity["isaac_urdf"]["sha256"]) == 64


def test_runtime_manifest_keeps_usd_optional_for_urdf_tasks(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("G1_USD_PATH", raising=False)
    monkeypatch.delenv("G1_ISAAC_URDF_PATH", raising=False)

    identity = _asset_identity(tmp_path)

    assert identity["isaac_urdf"] == {"path": None, "status": "not_configured"}
    assert identity["isaac_usd"] == {"path": None, "status": "not_configured"}
