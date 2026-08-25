from pathlib import Path

from backend.app.workers.stage_runner import StageExecutionContext, StageRunner, idempotency_key


def test_stage_runner_uses_stable_idempotency_key(tmp_path: Path) -> None:
    context = StageExecutionContext("validate_input", tmp_path / "attempt", "a" * 64, "processor.v1", "b" * 64, "g1.v1")
    assert context.key == idempotency_key(input_hash="a" * 64, processor_version="processor.v1", config_hash="b" * 64, robot_version="g1.v1")
    result = StageRunner().run(context, ["python", "-c", "print('ok')"])
    assert result.succeeded
    assert result.command == ("python", "-c", "print('ok')")

