from backend.app.application.reward_catalog import RewardConfigVersionStore, default_reward_config, validate_reward_config


def test_reward_parameter_schema_is_enforced() -> None:
    config = default_reward_config()
    config.terms[0].params["sigma"] = 100.0
    result = validate_reward_config(config)
    assert not result.valid
    assert any(issue.code == "REWARD_PARAM_OUT_OF_RANGE" for issue in result.issues)


def test_reward_versions_are_immutable_and_parented() -> None:
    store = RewardConfigVersionStore()
    first = store.create(default_reward_config())
    second = store.create(default_reward_config(), parent_version_id=first.version_id)
    assert first.version == 1
    assert second.version == 2
    assert second.parent_version_id == first.version_id
    assert second.config_sha256 != ""

