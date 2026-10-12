import jax
import numpy as np

from tiny_sima.config import SkillPPOConfig
from tiny_sima.envs.skills import make_vectorized_skill_environment
from tiny_sima.evaluation.goal_conditioned import (
    diagnose_goal_conditioned_skill,
    evaluate_goal_conditioned_skill,
)
from tiny_sima.executor.model import GoalConditionedActorCriticRNN
from tiny_sima.executor.ppo_common import (
    normalize_advantages_by_group,
    stratified_minibatch_indices,
)
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
        num_minibatches=2,
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
        "macro_success_rate",
        "skill/collect_wood/success_rate",
        "skill/collect_wood/feasible_timeout_rate",
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
    assert 0.0 <= evaluation["feasible_timeout_rate"] <= 1.0
    assert 0.0 <= evaluation["precondition_lost_rate"] <= 1.0

    diagnostic = diagnose_goal_conditioned_skill(
        updated_runner.train_state.params,
        config,
        "collect_wood",
        num_scenarios=2,
        seed=5,
    )
    assert diagnostic["skill_structure"] == "turn_then_interact"
    assert diagnostic["initial"]["target"] == "face_target"
    assert diagnostic["after_oracle_turn"]["target"] == "do"
    assert 0.0 <= diagnostic["initial"]["top1_accuracy"] <= 1.0
    assert 0.0 <= diagnostic["estimated_stochastic_sequence_probability"] <= 1.0

    direct_diagnostic = diagnose_goal_conditioned_skill(
        updated_runner.train_state.params,
        config,
        "place_table",
        num_scenarios=2,
        seed=6,
    )
    assert direct_diagnostic["skill_structure"] == "direct"
    assert direct_diagnostic["initial"]["target"] == "place_table"
    assert direct_diagnostic["after_oracle_turn"] is None


def test_stratified_minibatches_include_every_skill() -> None:
    indices = stratified_minibatch_indices(
        jax.random.PRNGKey(0),
        num_groups=6,
        items_per_group=4,
        num_minibatches=2,
    )
    groups = np.asarray(indices) // 4

    assert indices.shape == (2, 12)
    for minibatch_groups in groups:
        assert np.bincount(minibatch_groups, minlength=6).tolist() == [2] * 6


def test_advantage_normalization_is_independent_per_skill() -> None:
    advantages = jax.numpy.asarray(
        [
            [1.0, 10.0],
            [3.0, 14.0],
        ]
    )
    skill_ids = jax.numpy.asarray(
        [
            [0, 1],
            [0, 1],
        ]
    )
    normalized = np.asarray(
        normalize_advantages_by_group(
            advantages,
            skill_ids,
            num_groups=2,
        )
    )

    np.testing.assert_allclose(normalized[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(normalized[:, 1], [-1.0, 1.0])
