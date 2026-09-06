from backend.app.application.reward_catalog import RewardConfigVersionStore, default_reward_config, validate_reward_config


def test_reward_parameter_schema_is_enforced() -> None:
    config = default_reward_config(robot_id="unitree_g1_29dof", task_id="g1_mimic")
    config.terms[0].params["sigma"] = 100.0
    result = validate_reward_config(config, robot_id="unitree_g1_29dof", task_id="g1_mimic")
    assert not result.valid
    assert any(issue.code == "REWARD_PARAM_OUT_OF_RANGE" for issue in result.issues)


def test_reward_versions_are_immutable_and_parented() -> None:
    store = RewardConfigVersionStore()
    config = default_reward_config(robot_id="unitree_g1_29dof", task_id="g1_mimic")
    first = store.create(config, robot_id="unitree_g1_29dof", task_id="g1_mimic")
    second = store.create(config, parent_version_id=first.version_id, robot_id="unitree_g1_29dof", task_id="g1_mimic")
    assert first.version == 1
    assert second.version == 2
    assert second.parent_version_id == first.version_id
    assert second.config_sha256 != ""


def test_reward_versions_can_be_loaded_by_hash_across_processes(tmp_path) -> None:
    config = default_reward_config(robot_id="unitree_g1_29dof", task_id="g1_mimic")
    path = tmp_path / "reward_configs.json"
    first_store = RewardConfigVersionStore(storage_path=path)
    first = first_store.create(config, robot_id="unitree_g1_29dof", task_id="g1_mimic")
    second_store = RewardConfigVersionStore(storage_path=path)
    loaded = second_store.get_by_sha256(first.config_sha256)
    assert loaded is not None
    assert loaded.config.model_dump(mode="json") == config.model_dump(mode="json")


def test_running_worker_refreshes_reward_versions_from_shared_store(tmp_path) -> None:
    config = default_reward_config(robot_id="unitree_g1_29dof", task_id="g1_mimic")
    path = tmp_path / "reward_configs.json"
    worker_store = RewardConfigVersionStore(storage_path=path)
    api_store = RewardConfigVersionStore(storage_path=path)
    created = api_store.create(config, robot_id="unitree_g1_29dof", task_id="g1_mimic")
    assert worker_store.get_by_sha256(created.config_sha256) is not None
