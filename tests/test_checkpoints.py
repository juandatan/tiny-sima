from pathlib import Path

import jax
import jax.numpy as jnp

from tiny_sima.config import PPOConfig
from tiny_sima.executor.checkpoints import load_checkpoint, save_checkpoint
from tiny_sima.executor.model import ActorCriticRNN, initialize_parameters


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = PPOConfig(
        total_timesteps=4,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        optimistic_reset_ratio=2,
        hidden_size=16,
    ).validate()
    model = ActorCriticRNN(num_actions=3, hidden_size=16)
    params = initialize_parameters(
        model,
        jax.random.PRNGKey(0),
        (8,),
    )

    checkpoint = save_checkpoint(
        tmp_path / "checkpoint",
        params,
        config,
        timesteps=4,
        metrics={"reward": 1.5},
    )
    restored, metadata = load_checkpoint(checkpoint, params)

    assert metadata["timesteps"] == 4
    assert metadata["metrics"]["reward"] == 1.5
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda left, right: jnp.array_equal(left, right),
            params,
            restored,
        )
    )
