-- P2 PostgreSQL 16 baseline.  The application repositories remain the
-- authoritative port; this migration is intentionally free of ORM metadata.
create extension if not exists pgcrypto;

create table if not exists users (
  id uuid primary key,
  email text,
  status text not null default 'active',
  created_at timestamptz not null default now()
);

create table if not exists projects (
  id uuid primary key,
  name text not null,
  owner_id uuid not null references users(id),
  status text not null default 'ACTIVE',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists project_members (
  project_id uuid not null references projects(id),
  user_id uuid not null references users(id),
  role text not null check (role in ('owner', 'editor', 'viewer')),
  created_at timestamptz not null default now(),
  primary key (project_id, user_id)
);

create table if not exists assets (
  id uuid primary key,
  project_id uuid not null references projects(id),
  kind text not null check (kind in ('video', 'motion', 'model')),
  display_name text not null,
  license_json jsonb not null,
  status text not null default 'ACTIVE',
  created_by uuid not null references users(id),
  created_at timestamptz not null default now()
);

create table if not exists asset_versions (
  id uuid primary key,
  asset_id uuid not null references assets(id),
  version integer not null,
  status text not null default 'UPLOADING',
  object_key text not null,
  original_filename text not null,
  content_type text,
  size_bytes bigint,
  sha256 char(64),
  created_at timestamptz not null default now(),
  validated_at timestamptz,
  rejection_code text,
  unique(asset_id, version),
  unique(sha256)
);

create table if not exists runs (
  id uuid primary key,
  project_id uuid not null references projects(id),
  status text not null,
  parent_run_id uuid references runs(id),
  created_by uuid not null references users(id),
  current_attempt_id uuid not null,
  manifest_json jsonb not null,
  manifest_sha256 char(64) not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists attempts (
  id uuid primary key,
  run_id uuid not null references runs(id),
  number integer not null,
  status text not null,
  started_at timestamptz,
  finished_at timestamptz,
  worker_id text,
  gpu_uuid text,
  exit_code integer,
  failure_code text,
  last_heartbeat_at timestamptz,
  unique(run_id, number)
);

do $$ begin
  if not exists (select 1 from pg_constraint where conname = 'runs_current_attempt_fk') then
    alter table runs add constraint runs_current_attempt_fk foreign key (current_attempt_id) references attempts(id) deferrable initially deferred;
  end if;
end $$;

create table if not exists log_events (
  run_id uuid not null references runs(id),
  attempt_id uuid not null references attempts(id),
  seq bigint not null,
  event_type text not null,
  stage text,
  level text not null default 'INFO',
  message text not null default '',
  payload_json jsonb not null default '{}',
  created_at timestamptz not null default now(),
  primary key(attempt_id, seq)
);

create table if not exists audit_events (
  id uuid primary key,
  project_id uuid references projects(id),
  actor_id uuid not null references users(id),
  action text not null,
  resource_type text not null,
  resource_id text not null,
  payload_json jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create table if not exists outbox_events (
  id uuid primary key,
  topic text not null,
  event_key text not null,
  payload_json jsonb not null,
  created_at timestamptz not null default now(),
  published_at timestamptz
);

create unique index if not exists asset_versions_sha256_idx on asset_versions(sha256) where sha256 is not null;
create index if not exists projects_owner_updated_idx on projects(owner_id, updated_at desc);
create index if not exists runs_project_created_idx on runs(project_id, created_at desc);
create index if not exists attempts_run_number_idx on attempts(run_id, number);
create index if not exists log_events_attempt_seq_idx on log_events(attempt_id, seq);
create index if not exists outbox_pending_idx on outbox_events(created_at) where published_at is null;
