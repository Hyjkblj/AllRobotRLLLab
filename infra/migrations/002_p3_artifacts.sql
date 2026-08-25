-- P3 immutable artifact index. Large payloads stay in object storage.
create table if not exists artifacts (
  id uuid primary key,
  run_id uuid not null references runs(id),
  attempt_id uuid not null references attempts(id),
  kind text not null,
  object_key text not null,
  sha256 char(64) not null,
  size_bytes bigint not null check (size_bytes >= 0),
  content_type text,
  created_at timestamptz not null default now(),
  unique(run_id, attempt_id, kind, sha256)
);

create index if not exists artifacts_run_created_idx on artifacts(run_id, created_at desc);
create index if not exists artifacts_attempt_idx on artifacts(attempt_id, created_at desc);
