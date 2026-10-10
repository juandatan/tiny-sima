from pathlib import Path

import pytest

from tiny_sima.config import PPOConfig, load_config, save_config


def test_default_config_is_valid() -> None:
    config = PPOConfig().validate()

    assert config.batch_size == 512
    assert config.num_updates == 16
    assert config.envs_per_minibatch == 4


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    expected = PPOConfig(seed=7, hidden_size=64).validate()

    save_config(expected, path)
    actual = load_config(path)

    assert actual == expected


def test_config_rejects_invalid_minibatches() -> None:
    with pytest.raises(ValueError, match="num_envs must be divisible"):
        PPOConfig(
            total_timesteps=640,
            num_envs=10,
            num_steps=32,
            num_minibatches=4,
        ).validate()


def test_config_rejects_unknown_override() -> None:
    with pytest.raises(ValueError, match="Unknown PPO configuration"):
        PPOConfig().with_overrides(not_a_field=1)


def test_config_rejects_invalid_optimistic_reset_ratio() -> None:
    with pytest.raises(ValueError, match="cannot exceed num_envs"):
        PPOConfig(
            num_envs=8,
            num_minibatches=4,
            optimistic_reset_ratio=16,
        ).validate()


def test_config_allows_disabling_optimistic_resets() -> None:
    config = PPOConfig(
        num_envs=8,
        num_minibatches=4,
        use_optimistic_resets=False,
        optimistic_reset_ratio=16,
    ).validate()

    assert not config.use_optimistic_resets
