import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tiny_sima.envs.skills import (
    LOCAL_NAVIGATION_SKILLS,
    SkillScenario,
    SkillScenarioConfig,
    SkillStep,
    make_skill_environment,
    make_vectorized_skill_environment,
    reset_skill_batch,
    skills_for_scenario,
)
from tiny_sima.evaluation.skills import (
    evaluate_skill_policy,
    scripted_skill_action,
)
from tiny_sima.goals.skills import (
    ALL_SKILL_IDS,
    SKILL_IDS,
    SKILL_REGISTRY,
    SkillId,
    skill_precondition,
)


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
    assert state.target_position.shape == (4, 2)
    assert state.approach_position.shape == (4, 2)


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


def test_vectorized_local_training_keeps_place_table_nearby() -> None:
    environment = make_vectorized_skill_environment(
        num_envs=12,
        horizon=32,
        scenario=SkillScenario.LOCAL_NAVIGATION,
        min_distance=3,
        max_distance=4,
        distractor_count=5,
    )

    scenarios = {
        skill_environment.goal.skill_id: skill_environment.scenario
        for skill_environment in environment.environments
    }
    for skill_id in skills_for_scenario(SkillScenario.LOCAL_NAVIGATION):
        assert scenarios[skill_id] == SkillScenarioConfig(
            SkillScenario.LOCAL_NAVIGATION,
            min_distance=3,
            max_distance=4,
            distractor_count=5,
        )
    assert scenarios[SkillId.PLACE_TABLE].kind is SkillScenario.NEARBY


@pytest.mark.parametrize("skill_id", LOCAL_NAVIGATION_SKILLS)
def test_local_navigation_scripted_policy_succeeds(
    skill_id: SkillId,
) -> None:
    result = evaluate_skill_policy(
        skill_id,
        policy="scripted",
        num_episodes=16,
        horizon=32,
        scenario=SkillScenario.LOCAL_NAVIGATION,
        min_distance=2,
        max_distance=4,
        distractor_count=6,
    )

    assert result["scenario"] == "local_navigation"
    assert result["success_rate"] == 1.0


def test_local_navigation_excludes_non_navigation_skills() -> None:
    assert skills_for_scenario(SkillScenario.LOCAL_NAVIGATION) == tuple(
        skill_id for skill_id in LOCAL_NAVIGATION_SKILLS if skill_id in SKILL_IDS
    )
    assert (
        skills_for_scenario(SkillScenario.LOCAL_NAVIGATION, ALL_SKILL_IDS)
        == LOCAL_NAVIGATION_SKILLS
    )
    with pytest.raises(ValueError, match="does not define"):
        make_skill_environment(
            SkillId.PLACE_TABLE,
            scenario=SkillScenario.LOCAL_NAVIGATION,
        )


def test_local_navigation_randomizes_target_and_adds_distractors() -> None:
    environment = make_skill_environment(
        SkillId.COLLECT_WOOD,
        horizon=32,
        scenario=SkillScenario.LOCAL_NAVIGATION,
        min_distance=3,
        max_distance=4,
        distractor_count=6,
    )
    resets = [
        environment.reset(jax.random.PRNGKey(seed)).state
        for seed in range(4)
    ]
    target_positions = {
        tuple(np.asarray(reset.target_position).tolist()) for reset in resets
    }

    assert len(target_positions) > 1
    for reset in resets:
        player = np.asarray(reset.env_state.player_position)
        target = np.asarray(reset.target_position)
        distance = int(np.abs(target - player).sum())
        assert 3 <= distance <= 4
        assert int(reset.env_state.map[target[0], target[1]]) == BlockType.TREE.value
        local_map = np.asarray(reset.env_state.map)[29:36, 28:37]
        non_grass = np.count_nonzero(local_map != BlockType.GRASS.value)
        assert non_grass >= 7


def test_skill_scenario_config_validates_bounds() -> None:
    with pytest.raises(ValueError, match="min_distance"):
        SkillScenarioConfig(min_distance=0)
    with pytest.raises(ValueError, match="max_distance"):
        SkillScenarioConfig(min_distance=4, max_distance=3)
    with pytest.raises(ValueError, match="distractor_count"):
        SkillScenarioConfig(distractor_count=-1)
