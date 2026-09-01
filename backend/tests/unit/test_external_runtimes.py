import json
import os
import sys
from pathlib import Path

import numpy as np

from backend.app.runtime.gmr_runner import GmrRunner
from backend.app.runtime.mujoco_kinematics import MuJoCoKinematicsCompiler
from backend.app.runtime.registry import RuntimeRegistry


def test_runtime_registry_registers_embedded_source_by_content_hash(tmp_path: Path) -> None:
    runtime = tmp_path / "gmr"
    runtime.mkdir()
    (runtime / "README.md").write_text("fixture", encoding="utf-8")
    registry = RuntimeRegistry(registration_path=tmp_path / "registrations.json")
    check = registry.register("gmr", path=runtime)
    assert check.available
    stored = json.loads((tmp_path / "registrations.json").read_text(encoding="utf-8"))
    assert stored["gmr"]["revision"].startswith("content:")


def test_gmr_runner_normalizes_fixture_output(tmp_path: Path, monkeypatch) -> None:
    runtime = tmp_path / "gmr"
    runtime.mkdir()
    registration = tmp_path / "registrations.json"
    registry = RuntimeRegistry(registration_path=registration)
    registry.register("gmr", path=runtime, python=sys.executable)
    source = tmp_path / "input.pt"
    source.write_bytes(b"fixture")
    script = tmp_path / "emit.py"
    script.write_text("import os,pickle,numpy as np; n=15; pickle.dump({'dof_pos':np.zeros((n,29)),'root_pos':np.zeros((n,3)),'root_rot':np.tile([0,0,0,1],(n,1)),'fps':30},open(os.environ['GMR_OUTPUT'],'wb'))", encoding="utf-8")
    monkeypatch.setenv("GMR_COMMAND", f'"{sys.executable}" "{script}"')
    # The command override is intentionally independent of the upstream path.
    output, _ = GmrRunner(registry=registry, workspace=tmp_path / "work").run(source_path=source, output_dir=tmp_path / "work" / "run")
    assert output.is_file()
    assert (tmp_path / "work" / "run" / "retarget_motion.npz").is_file()


def test_mujoco_compiler_explicitly_reports_missing_model(tmp_path: Path) -> None:
    compiler = MuJoCoKinematicsCompiler(model_path=tmp_path / "missing.xml", body_names=("root",), joint_names=("joint",), allow_approximation=False)
    try:
        compiler.compile(joint_pos=np.zeros((1, 1)), root_pos=np.zeros((1, 3)), root_rot=np.array([[0, 0, 0, 1.0]]), fps=30)
    except Exception as exc:
        assert getattr(exc, "code", "") == "KINEMATICS_MODEL_NOT_FOUND"
    else:
        raise AssertionError("missing MuJoCo model was accepted")
