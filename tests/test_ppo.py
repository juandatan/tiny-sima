import jax
import numpy as np

from tiny_sima.config import PPOConfig
from tiny_sima.envs.craftax import make_environment
from tiny_sima.executor.model import ActorCriticRNN
from tiny_sima.executor.ppo import initialize_runner, make_update


def test_single_ppo_update() -> None:
    config = PPOConfig(
        total_timesteps=4,
        num_envs=2,
        num_steps=2,
        max_episode_steps=8,
        update_epochs=1,
        num_minibatches=1,
        hidden_size=16,
        log_every=1,
    ).validate()
    bundle = make_environment(
        config.env_name,
        max_episode_steps=config.max_episode_steps,
    )
    model = ActorCriticRNN(
        num_actions=bundle.spec.num_actions,
        hidden_size=config.hidden_size,
    )
    runner = initialize_runner(config, bundle, model)

    updated_runner, metrics = make_update(config, bundle, model)(runner)
    jax.block_until_ready(metrics)

    assert int(updated_runner.train_state.step) == 1
    assert set(metrics) >= {
        "loss",
        "actor_loss",
        "value_loss",
        "entropy",
        "reward_mean",
    }
    assert all(np.isfinite(np.asarray(value)) for value in metrics.values())
