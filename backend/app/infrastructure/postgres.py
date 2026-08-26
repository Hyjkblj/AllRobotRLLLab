"""PostgreSQL connectivity and migration utilities.

Repository implementations use the domain ports; this module owns connection
setup and schema initialization so API imports never open a database eagerly.
"""

from __future__ import annotations

from pathlib import Path


class PostgresDatabase:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def connect(self):
        import psycopg

        return psycopg.connect(self.dsn)

    def health(self) -> bool:
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute("select 1")
            return cursor.fetchone() == (1,)

    def apply_migration(self, path: Path) -> None:
        sql = path.resolve().read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.execute(sql)
            connection.commit()

    def apply_migrations(self, directory: Path) -> list[str]:
        """Apply all ordered SQL migrations and return applied file names.

        Migrations are intentionally plain SQL so deployment can run this
        method once before starting API/workers without importing ORM state.
        Each file is idempotent and commits independently.
        """
        paths = sorted(directory.resolve().glob("*.sql"))
        if not paths:
            return []
        with self.connect() as connection, connection.cursor() as cursor:
            # Serialize deploys and keep a durable record so a reused
            # PostgreSQL volume receives new migrations exactly once.
            cursor.execute("select pg_advisory_xact_lock(hashtext('allrobotrl:migrations'))")
            cursor.execute("create table if not exists schema_migrations (version text primary key, applied_at timestamptz not null default now())")
            cursor.execute("select version from schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}
            completed: list[str] = []
            for path in paths:
                if path.name in applied:
                    completed.append(path.name)
                    continue
                cursor.execute(path.read_text(encoding="utf-8"))
                cursor.execute("insert into schema_migrations(version) values (%s)", (path.name,))
                completed.append(path.name)
            connection.commit()
        return completed


__all__ = ["PostgresDatabase"]
