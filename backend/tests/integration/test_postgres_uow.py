"""PostgreSQL contract smoke; enabled when TEST_DATABASE_URL is provided."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from backend.app.application.run_service import RunService
from backend.app.domain.contracts import Actor, P3RunState, TrainingConfig
from backend.app.infrastructure.postgres import PostgresDatabase
from backend.app.infrastructure.postgres_uow import PostgresUnitOfWork


pytestmark = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL is not configured")


def test_postgres_uow_round_trip() -> None:
    dsn = os.environ["TEST_DATABASE_URL"]
    root = Path(__file__).resolve().parents[3]
    PostgresDatabase(dsn).apply_migrations(root / "infra" / "migrations")
    service = RunService(PostgresUnitOfWork(dsn))
    actor = Actor(user_id=f"pytest-{uuid.uuid4()}")
    project = service.create_project(name="postgres contract", actor=actor)
    run, attempt, replayed = service.create_run(actor=actor, project_id=project.project_id, robot={"robot_id": "unitree_g1_29dof"}, motion={"train_motion_sha256": "a" * 64}, reward_config_sha256="b" * 64, training_config_sha256="c" * 64, idempotency_key=f"pytest-{uuid.uuid4()}")
    assert not replayed
    loaded, attempts = service.get_run(run_id=run.run_id, actor=actor)
    assert loaded.run_id == run.run_id
    assert attempts[0].attempt_id == attempt.attempt_id
    event = service.append_event(run_id=run.run_id, event_type="log", stage="pytest", message="postgres")
    assert service.list_events(run_id=run.run_id, actor=actor)[-1].seq == event.seq

    state = P3RunState(run_id=run.run_id, attempt_id=attempt.attempt_id, training_config=TrainingConfig(motion_asset_version_id="motion-1"), updated_at="2026-01-01T00:00:00+00:00")
    with PostgresUnitOfWork(dsn) as uow:
        uow.p3_states.upsert(state)
    with PostgresUnitOfWork(dsn) as uow:
        restored = uow.p3_states.get(run.run_id)
    assert restored is not None
    assert restored.training_config is not None
    assert restored.training_config.motion_asset_version_id == "motion-1"
