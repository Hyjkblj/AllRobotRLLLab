-- External user identities remain stable strings at the API boundary while
-- existing tables keep UUID foreign keys.
alter table users add column if not exists external_id text;
create unique index if not exists users_external_id_idx on users(external_id) where external_id is not null;

alter table attempts add column if not exists created_at timestamptz not null default now();

create table if not exists run_idempotency (
  idempotency_key text primary key,
  run_id uuid not null unique references runs(id),
  created_at timestamptz not null default now()
);
