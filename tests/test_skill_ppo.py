import jax
import numpy as np

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.skills import make_vectorized_skill_environment
from tiny_sima.evaluation.goal_conditioned import (
    evaluate_goal_conditioned_skill,
)
from tiny_sima.executor.model import GoalConditionedActorCriticRNN
from tiny_sima.executor.skill_ppo import (
    initialize_skill_runner,
    make_skill_update,
)


def test_single_goal_conditioned_ppo_update() -> None:
    config = SkillPPOConfig(
        total_timesteps=24,
        num_envs=12,
        num_steps=2,
        max_episode_steps=8,
        use_optimistic_resets=False,
        optimistic_reset_ratio=6,
        update_epochs=1,
        num_minibatches=3,
        hidden_size=16,
        skill_embedding_size=4,
        skill_horizon=4,
        log_every=1,
        monitor_gpu=False,
    ).validate()
    environment = make_vectorized_skill_environment(
        num_envs=config.num_envs,
        horizon=config.skill_horizon,
    )
    model = GoalConditionedActorCriticRNN(
        num_actions=environment.spec.num_actions,
        num_skills=environment.num_skills,
        hidden_size=config.hidden_size,
        skill_embedding_size=config.skill_embedding_size,
    )
    runner = initialize_skill_runner(config, environment, model)

    updated_runner, metrics = make_skill_update(
        config,
        environment,
        model,
    )(runner)
    jax.block_until_ready(metrics)

    assert int(updated_runner.train_state.step) == config.num_minibatches
    assert set(metrics) >= {
        "loss",
        "actor_loss",
        "value_loss",
        "entropy",
        "reward_mean",
        "success_rate",
    }
    assert all(np.isfinite(np.asarray(value)) for value in metrics.values())

    evaluation = evaluate_goal_conditioned_skill(
        updated_runner.train_state.params,
        config,
        "collect_wood",
        num_episodes=2,
        seed=4,
    )
    assert 0.0 <= evaluation["success_rate"] <= 1.0
    assert 1.0 <= evaluation["mean_episode_length"] <= config.skill_horizon
