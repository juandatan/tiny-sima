import jax
import numpy as np

from tiny_sima.config import PPOConfig
from tiny_sima.envs.craftax import make_vectorized_environment
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
        optimistic_reset_ratio=2,
        hidden_size=16,
        log_every=1,
    ).validate()
    vector_env = make_vectorized_environment(
        config.env_name,
        num_envs=config.num_envs,
        max_episode_steps=config.max_episode_steps,
        use_optimistic_resets=config.use_optimistic_resets,
        optimistic_reset_ratio=config.optimistic_reset_ratio,
    )
    model = ActorCriticRNN(
        num_actions=vector_env.spec.num_actions,
        hidden_size=config.hidden_size,
    )
    runner = initialize_runner(config, vector_env, model)

    updated_runner, metrics = make_update(config, vector_env, model)(runner)
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
