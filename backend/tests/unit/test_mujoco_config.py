import json
from pathlib import Path

import pytest

from backend.app.runtime.mujoco_config import resolve_mujoco_model_config


def test_mujoco_model_config_resolves_robot_spec_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "h1.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    spec = tmp_path / "robot_spec.json"
    spec.write_text(json.dumps({"robot_id": "h1", "assets": {"mujoco_xml_uri": "h1.xml"}}), encoding="utf-8")
    monkeypatch.setenv("MOTIONLAB_ROBOT_SPEC", str(spec))
    monkeypatch.delenv("MOTIONLAB_MODEL_PATH", raising=False)
    monkeypatch.delenv("MOTIONLAB_ROBOT_ID", raising=False)
    monkeypatch.setenv("MOTIONLAB_ROOT_BODY", "torso")
    monkeypatch.setenv("MOTIONLAB_ROOT_HEIGHT", "1.06")
    config = resolve_mujoco_model_config(repository_root=tmp_path)
    assert config.robot_id == "h1"
    assert config.model_path == model.resolve()
    assert config.model_name == "h1"
    assert config.root_body == "torso"
    assert config.root_height == 1.06


def test_mujoco_model_config_uses_spec_root_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "h1.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    spec = tmp_path / "robot_spec.json"
    spec.write_text(json.dumps({"robot_id": "h1", "assets": {"mujoco_xml_uri": "h1.xml"}, "body_names": ["torso"], "default_root_height": 1.05}), encoding="utf-8")
    monkeypatch.setenv("MOTIONLAB_ROBOT_SPEC", str(spec))
    monkeypatch.delenv("MOTIONLAB_MODEL_PATH", raising=False)
    monkeypatch.delenv("MOTIONLAB_ROOT_BODY", raising=False)
    monkeypatch.delenv("MOTIONLAB_ROOT_HEIGHT", raising=False)
    config = resolve_mujoco_model_config(repository_root=tmp_path)
    assert config.root_body == "torso"
    assert config.root_height == 1.05


def test_mujoco_model_config_rejects_missing_explicit_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOTIONLAB_MODEL_PATH", str(tmp_path / "missing.xml"))
    monkeypatch.delenv("MOTIONLAB_ROBOT_SPEC", raising=False)
    with pytest.raises(FileNotFoundError, match="MuJoCo model was not found"):
        resolve_mujoco_model_config(repository_root=tmp_path)


def test_mujoco_model_config_never_uses_g1_fallback_for_another_robot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOTIONLAB_ROBOT_ID", "unitree_h1_19dof")
    monkeypatch.delenv("MOTIONLAB_MODEL_PATH", raising=False)
    monkeypatch.delenv("MOTIONLAB_ROBOT_SPEC", raising=False)
    with pytest.raises(FileNotFoundError, match="not configured for unitree_h1_19dof"):
        resolve_mujoco_model_config(repository_root=tmp_path)
