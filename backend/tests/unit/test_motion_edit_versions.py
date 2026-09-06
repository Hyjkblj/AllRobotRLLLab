from backend.app.application.motion_editor import MotionEditVersionStore
from backend.app.domain.contracts import MotionEditConfig


def test_motion_edit_versions_survive_store_reinitialization(tmp_path) -> None:
    path = tmp_path / "motion_edits.json"
    config = MotionEditConfig(source_motion_version_id="motion-v1", robot_id="unitree_g1_29dof")
    first_store = MotionEditVersionStore(storage_path=path)
    first = first_store.create(config)

    second_store = MotionEditVersionStore(storage_path=path)
    loaded = second_store.get(first.version_id)
    assert loaded is not None
    assert loaded.config_sha256 == first.config_sha256
    assert second_store.list_for_source("motion-v1")[0].version == 1


def test_motion_edit_store_refreshes_before_allocating_next_version(tmp_path) -> None:
    path = tmp_path / "motion_edits.json"
    config = MotionEditConfig(source_motion_version_id="motion-v1", robot_id="unitree_g1_29dof")
    first_store = MotionEditVersionStore(storage_path=path)
    first_store.create(config)
    second_store = MotionEditVersionStore(storage_path=path)
    second_store.create(config)

    third = first_store.create(config)
    assert third.version == 3
