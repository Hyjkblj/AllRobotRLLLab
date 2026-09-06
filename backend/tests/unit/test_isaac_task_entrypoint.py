from pathlib import Path
import pytest

from apps.isaac_tasks.entrypoint import load_task_modules, register_task, resolve_task
from apps.isaac_tasks.delegated import DelegatedIsaacTask, TaskImplementationError


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


def test_delegated_task_requires_explicit_isaac_implementation(monkeypatch) -> None:
    task = DelegatedIsaacTask("delegated-test", "DELEGATED_TEST_MODULE")
    monkeypatch.delenv("DELEGATED_TEST_MODULE", raising=False)
    with pytest.raises(TaskImplementationError, match="must point"):
        task.train(task_id="delegated-test", manifest=Path("manifest.json"), output_dir=Path("out"))
