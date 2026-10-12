import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action

from tiny_sima.envs.skills import (
    SkillStep,
    make_skill_environment,
    make_vectorized_skill_environment,
    reset_skill_batch,
)
from tiny_sima.evaluation.skills import (
    evaluate_skill_policy,
    scripted_skill_action,
)
from tiny_sima.goals.skills import SKILL_REGISTRY, SkillId, skill_precondition


@pytest.mark.parametrize("skill_id", list(SKILL_REGISTRY))
def test_skill_scenarios_are_feasible_and_scripted_policy_succeeds(
    skill_id: SkillId,
) -> None:
    environment = make_skill_environment(skill_id, horizon=8)
    _, state = environment.reset(jax.random.PRNGKey(0))

    assert bool(np.asarray(skill_precondition(environment.goal, state.env_state)))

    result: SkillStep | None = None
    reward = 0.0
    for step in range(2):
        action = scripted_skill_action(environment, state)
        result = environment.step(
            jax.random.PRNGKey(step + 1),
            state,
            action,
        )
        state = result.state
        reward += float(np.asarray(result.reward))
        if bool(np.asarray(result.done)):
            break

    assert result is not None
    assert bool(np.asarray(result.done))
    assert bool(np.asarray(result.info["skill_success"]))
    assert reward == 1.0


def test_skill_episode_times_out_without_completion() -> None:
    environment = make_skill_environment(SkillId.COLLECT_WOOD, horizon=2)
    _, state = environment.reset(jax.random.PRNGKey(0))

    result: SkillStep | None = None
    for step in range(2):
        result = environment.step(
            jax.random.PRNGKey(step + 1),
            state,
            jnp.asarray(Action.NOOP.value),
        )
        state = result.state

    assert result is not None
    assert bool(np.asarray(result.done))
    assert bool(np.asarray(result.info["skill_timeout"]))
    assert not bool(np.asarray(result.info["skill_success"]))
    assert float(np.asarray(result.reward)) == 0.0


def test_wrong_crafting_action_can_invalidate_requested_skill() -> None:
    environment = make_skill_environment(
        SkillId.CRAFT_WOOD_PICKAXE,
        horizon=4,
    )
    _, state = environment.reset(jax.random.PRNGKey(0))

    result = environment.step(
        jax.random.PRNGKey(1),
        state,
        jnp.asarray(Action.MAKE_WOOD_SWORD.value),
    )

    assert bool(np.asarray(result.info["skill_precondition"]))
    assert not bool(np.asarray(result.info["skill_precondition_after"]))
    assert not bool(np.asarray(result.info["skill_success"]))


def test_skill_batch_reset_shapes() -> None:
    environment = make_skill_environment(SkillId.DRINK_WATER)
    observation, state = reset_skill_batch(
        environment,
        jax.random.PRNGKey(0),
        num_envs=4,
    )

    assert observation.shape == (4, *environment.bundle.spec.observation_shape)
    assert state.steps.shape == (4,)
    assert state.target_direction.shape == (4,)


def test_scripted_skill_baseline_reaches_full_success() -> None:
    result = evaluate_skill_policy(
        SkillId.COLLECT_WOOD,
        policy="scripted",
        num_episodes=8,
        horizon=4,
    )

    assert result["success_rate"] == 1.0
    assert result["mean_return"] == 1.0


def test_random_skill_baseline_returns_valid_metrics() -> None:
    result = evaluate_skill_policy(
        SkillId.CRAFT_WOOD_PICKAXE,
        policy="random",
        num_episodes=8,
        horizon=4,
    )

    assert 0.0 <= result["success_rate"] <= 1.0
    assert 1.0 <= result["mean_episode_length"] <= 4.0


def test_vectorized_skill_environment_is_balanced_and_auto_resets() -> None:
    environment = make_vectorized_skill_environment(num_envs=12, horizon=4)
    reset = environment.reset(jax.random.PRNGKey(0))

    assert reset.observation.shape == (
        12,
        *environment.spec.observation_shape,
    )
    assert np.bincount(np.asarray(reset.skill_id), minlength=6).tolist() == [2] * 6

    action = jnp.zeros((12,), dtype=jnp.int32)
    result = environment.step(jax.random.PRNGKey(1), reset.state, action)

    assert result.observation.shape == reset.observation.shape
    assert result.state.steps.shape == (12,)
    assert result.reward.shape == (12,)
    assert result.done.shape == (12,)


def test_vectorized_skill_environment_requires_balanced_groups() -> None:
    with pytest.raises(ValueError, match="divisible"):
        make_vectorized_skill_environment(num_envs=8)
