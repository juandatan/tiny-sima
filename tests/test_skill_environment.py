import jax
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action

from tiny_sima.envs.skills import (
    make_skill_environment,
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

    done = False
    reward = 0.0
    for step in range(2):
        action = scripted_skill_action(environment, state)
        _, state, reward_array, done_array, info = environment.step(
            jax.random.PRNGKey(step + 1),
            state,
            action,
        )
        reward += float(np.asarray(reward_array))
        done = bool(np.asarray(done_array))
        if done:
            break

    assert done
    assert bool(np.asarray(info["skill_success"]))
    assert reward == 1.0


def test_skill_episode_times_out_without_completion() -> None:
    environment = make_skill_environment(SkillId.COLLECT_WOOD, horizon=2)
    _, state = environment.reset(jax.random.PRNGKey(0))

    for step in range(2):
        _, state, reward, done, info = environment.step(
            jax.random.PRNGKey(step + 1),
            state,
            np.asarray(Action.NOOP.value),
        )

    assert bool(np.asarray(done))
    assert bool(np.asarray(info["skill_timeout"]))
    assert not bool(np.asarray(info["skill_success"]))
    assert float(np.asarray(reward)) == 0.0


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
