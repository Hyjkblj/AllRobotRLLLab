-- Durable P3 context shared by API and Celery worker processes.
create table if not exists p3_run_states (
  run_id uuid primary key references runs(id),
  attempt_id uuid not null references attempts(id),
  training_config_json jsonb,
  checkpoint_json jsonb,
  export_metadata_json jsonb,
  export_files_json jsonb not null default '[]',
  bundle_json jsonb,
  sim2sim_report_json jsonb,
  artifact_ids_json jsonb not null default '[]',
  updated_at timestamptz not null default now()
);

create index if not exists p3_run_states_attempt_idx on p3_run_states(attempt_id);
