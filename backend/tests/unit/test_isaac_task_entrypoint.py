from pathlib import Path
import ast
import pytest

from apps.isaac_tasks.entrypoint import load_task_modules, register_task, resolve_task
from apps.isaac_tasks.delegated import DelegatedIsaacTask, TaskImplementationError
from apps.isaac_tasks.g1_native import G1NativeIsaacTask


class _FixtureTask:
    def train(self, *, task_id: str, manifest: Path, output_dir: Path) -> int:
        return 0

    def play(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int:
        return 0

    def export(self, *, task_id: str, checkpoint: Path, output_dir: Path) -> int:
        return 0


def test_isaac_task_registry_resolves_registered_implementation() -> None:
    task_id = "fixture_entrypoint_task"
    register_task(task_id, _FixtureTask())
    assert resolve_task(task_id).__class__ is _FixtureTask


def test_empty_isaac_task_module_list_is_dependency_free() -> None:
    assert load_task_modules([]) == ()


def test_g1_task_uses_platform_native_implementation_by_default(monkeypatch) -> None:
    monkeypatch.delenv("G1_ISAAC_TASK_MODULE", raising=False)
    task = DelegatedIsaacTask("g1_mimic", "G1_ISAAC_TASK_MODULE", "apps.isaac_tasks.g1_native")
    assert isinstance(task._implementation(), G1NativeIsaacTask)


def test_native_g1_runtime_does_not_import_unitree_rl_lab() -> None:
    source_path = Path(__file__).resolve().parents[3] / "apps" / "isaac_tasks" / "g1_runtime.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "unitree_rl_lab" or name.startswith("unitree_rl_lab.") for name in imported)


def test_loading_builtin_modules_is_idempotent() -> None:
    first = load_task_modules(["apps.isaac_tasks.g1_mimic"])
    second = load_task_modules(["apps.isaac_tasks.g1_mimic"])
    assert first == second == ("apps.isaac_tasks.g1_mimic",)


def test_delegated_task_requires_explicit_isaac_implementation(monkeypatch) -> None:
    task = DelegatedIsaacTask("delegated-test", "DELEGATED_TEST_MODULE")
    monkeypatch.delenv("DELEGATED_TEST_MODULE", raising=False)
    with pytest.raises(TaskImplementationError, match="must point"):
        task.train(task_id="delegated-test", manifest=Path("manifest.json"), output_dir=Path("out"))
