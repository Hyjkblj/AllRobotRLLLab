# Contract snapshots

The canonical Python definitions for the P0 slice live in
`backend/app/domain/contracts.py` and are exposed through
`packages/contracts`. CI can materialize JSON Schema snapshots from
`TrainingConfig.model_json_schema()` and the other contract models during the
contract-publishing step. Generated snapshots belong in this directory; hand
editing them is intentionally avoided.

