from pathlib import Path

import scripts.collect_runtime_manifest as runtime_manifest
from scripts.collect_runtime_manifest import _asset_identity, collect


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


def test_collect_runtime_manifest_includes_enabled_robot_assets(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_ADAPTER_MODULES", "adapters.unitree_h1_19dof")
    manifest = collect(Path(__file__).resolve().parents[3], profile="api")
    assert "unitree_h1_19dof" in manifest["robot_assets"]
    assert manifest["robot_assets"]["unitree_h1_19dof"]["mujoco_xml_uri"]["exists"] is True


def test_source_hash_excludes_generated_artifacts_and_binary_outputs(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "runtime.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "training.log").write_text("large log\n", encoding="utf-8")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "policy.pt").write_bytes(b"weights-v1")
    first = runtime_manifest._source_hash(tmp_path)

    (tmp_path / "logs" / "training.log").write_text("different log\n", encoding="utf-8")
    (tmp_path / "checkpoints" / "policy.pt").write_bytes(b"weights-v2")
    assert runtime_manifest._source_hash(tmp_path) == first

    (tmp_path / "src" / "runtime.py").write_text("print('changed')\n", encoding="utf-8")
    assert runtime_manifest._source_hash(tmp_path) != first


def test_collect_runtime_manifest_uses_git_sha_without_source_hash(monkeypatch, tmp_path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "logs").mkdir()
    (runtime / "logs" / "training.log").write_text("generated\n", encoding="utf-8")
    monkeypatch.setenv("GMR_PATH", str(runtime))
    monkeypatch.setattr(runtime_manifest, "_git_revision", lambda path: "abc123")
    monkeypatch.setenv("ROBOT_ADAPTER_MODULES", "")

    manifest = collect(tmp_path, profile="api")

    assert manifest["external"]["gmr"]["git_sha"] == "abc123"
    assert manifest["external"]["gmr"]["source_sha256"] is None
