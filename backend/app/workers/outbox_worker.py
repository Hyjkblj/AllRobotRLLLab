"""Publish committed PostgreSQL outbox events to Celery."""

from __future__ import annotations

import argparse
import time

from backend.app.application.outbox_dispatcher import OutboxDispatcher
from backend.app.config.settings import settings
from backend.app.infrastructure.postgres_uow import PostgresUnitOfWork
from backend.app.infrastructure.queue import CeleryTaskDispatcher


def run(*, poll_interval: float = 1.0, batch_size: int = 100, once: bool = False) -> int:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for the outbox dispatcher")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required for the outbox dispatcher")
    bridge = OutboxDispatcher(PostgresUnitOfWork(settings.database_url), CeleryTaskDispatcher(settings.redis_url))
    while True:
        published = bridge.dispatch(limit=batch_size)
        if once:
            return published
        if published == 0:
            time.sleep(max(0.1, poll_interval))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run(poll_interval=args.poll_interval, batch_size=max(1, args.batch_size), once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
