from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tiny_sima.config import load_skill_config
from tiny_sima.envs.skills import (
    LOCAL_NAVIGATION_SKILLS,
    ScenarioRandomization,
    SkillScenario,
    _lowest_sleep_energy,
    make_skill_environment,
    make_vectorized_skill_environment,
    scenario_randomization_from_config,
)
from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_PICKAXE_CHAIN,
    _batched_skill_metric,
    _batched_skill_precondition,
    _scripted_action,
    chain_environment,
)
from tiny_sima.evaluation.open_ended import OPEN_ENDED_CHAIN
from tiny_sima.evaluation.skills import evaluate_skill_policy
from tiny_sima.goals import (
    ALL_SKILL_IDS,
    SKILL_ACHIEVEMENTS,
    STONE_PICKAXE_SKILL_IDS,
    SkillGoal,
    SkillId,
    StateRequirement,
    achievement_field,
    executor_skill_ids,
    skill_metric,
    skill_precondition,
)
from tiny_sima.goals.reachability import skill_interaction_distance, sleep_steps
from tiny_sima.memory import EpisodeMemory
from tiny_sima.planner.goal_selector import VITAL_SUBGOAL_FIELDS
from tiny_sima.planner.model_learning import (
    INDUCTION_FIELDS,
    gain_fields,
    induction_fields,
)
from tiny_sima.planner.subgoals import SUBGOAL_FIELDS, open_ended_subgoal_fields
from tiny_sima.planner.symbolic import (
    PLANNER_FIELDS,
    REGISTRY_MODEL,
    read_field,
    registry_model,
    shortest_plan,
)

NEW_SKILLS = ALL_SKILL_IDS[len(STONE_PICKAXE_SKILL_IDS) :]
# Skills a short scripted episode always completes: a fixed target and one
# interaction. Saplings drop one time in ten, a cow takes three hits and
# wanders, and sleep takes eleven steps per missing energy point.
DETERMINISTIC_SKILLS = tuple(
    skill
    for skill in NEW_SKILLS
    if skill not in {SkillId.COLLECT_SAPLING, SkillId.EAT_COW, SkillId.SLEEP}
)


def test_new_skills_append_after_the_stone_pickaxe_catalog() -> None:
    assert len(ALL_SKILL_IDS) == 16
    assert ALL_SKILL_IDS[:7] == STONE_PICKAXE_SKILL_IDS
    assert STONE_PICKAXE_SKILL_IDS[-1] is SkillId.CRAFT_STONE_PICKAXE
    assert executor_skill_ids(16) == ALL_SKILL_IDS
    assert set(SKILL_ACHIEVEMENTS) == set(ALL_SKILL_IDS)
    # The evaluations still offer the catalog the recorded executors know.
    assert WOOD_TO_STONE_PICKAXE_CHAIN.skills == STONE_PICKAXE_SKILL_IDS
    assert OPEN_ENDED_CHAIN.skills == STONE_PICKAXE_SKILL_IDS
    assert [skill for skill, _ in REGISTRY_MODEL] == list(STONE_PICKAXE_SKILL_IDS)


@pytest.mark.parametrize("skill_id", NEW_SKILLS)
def test_new_skill_scenarios_start_feasible(skill_id: SkillId) -> None:
    for scenario in (SkillScenario.NEARBY, SkillScenario.RANDOMIZED):
        environment = make_skill_environment(skill_id, horizon=64, scenario=scenario)
        for seed in range(4):
            _, state = environment.reset(jax.random.PRNGKey(seed))
            assert bool(skill_precondition(environment.goal, state.env_state))
            assert not bool(state.env_state.is_sleeping)


@pytest.mark.parametrize("skill_id", DETERMINISTIC_SKILLS)
def test_scripted_policy_completes_the_deterministic_new_skills(
    skill_id: SkillId,
) -> None:
    result = evaluate_skill_policy(
        skill_id, policy="scripted", num_episodes=8, horizon=8
    )
    assert result["success_rate"] == 1.0


def test_scripted_policy_completes_the_stochastic_new_skills() -> None:
    for skill_id, floor in (
        (SkillId.COLLECT_SAPLING, 0.75),
        (SkillId.SLEEP, 1.0),
        # The scripted policy keeps facing the cow's starting tile, so it
        # fails whenever the cow wanders off before its third hit.
        (SkillId.EAT_COW, 0.25),
    ):
        result = evaluate_skill_policy(
            skill_id, policy="scripted", num_episodes=32, horizon=64
        )
        assert result["success_rate"] >= floor, (skill_id, result)


def test_eat_cow_scenario_places_one_live_cow_on_the_target() -> None:
    environment = make_skill_environment(SkillId.EAT_COW, horizon=64)
    _, state = environment.reset(jax.random.PRNGKey(0))
    cows = state.env_state.cows
    target = np.asarray(state.target_position)

    assert int(np.count_nonzero(cows.mask)) == 1
    np.testing.assert_array_equal(np.asarray(cows.position[0]), target)
    assert bool(state.env_state.mob_map[target[0], target[1]])
    assert int(state.env_state.map[target[0], target[1]]) == BlockType.GRASS.value
    assert int(state.env_state.player_food) < 9


def test_sleep_succeeds_on_waking_rested() -> None:
    environment = make_skill_environment(SkillId.SLEEP, horizon=64)
    _, state = environment.reset(jax.random.PRNGKey(0))
    expected = int(sleep_steps(state.env_state))
    assert expected == int(
        skill_interaction_distance(environment.goal, state.env_state)
    )
    assert int(skill_metric(SkillId.SLEEP, state.env_state)) == 0

    for step in range(expected):
        result = environment.step(
            jax.random.PRNGKey(step), state, jnp.asarray(Action.SLEEP.value)
        )
        state = result.state
        assert bool(result.done) == (step == expected - 1)
    assert bool(result.info["skill_success"])
    assert bool(state.env_state.achievements[SKILL_ACHIEVEMENTS[SkillId.SLEEP].value])


def test_randomized_sleep_energy_fits_the_horizon() -> None:
    assert _lowest_sleep_energy(SkillGoal(SkillId.SLEEP, horizon=64)) == 4
    assert _lowest_sleep_energy(SkillGoal(SkillId.SLEEP, horizon=16)) == 8
    environment = make_skill_environment(
        SkillId.SLEEP,
        horizon=64,
        scenario=SkillScenario.RANDOMIZED,
        randomization=ScenarioRandomization(),
    )
    for seed in range(16):
        _, state = environment.reset(jax.random.PRNGKey(seed))
        assert 4 <= int(state.env_state.player_energy) <= 8
        assert int(sleep_steps(state.env_state)) <= 64


def test_sixteen_skill_config_builds_a_balanced_environment() -> None:
    config = load_skill_config("configs/skill_ppo_16_skill_randomized_debug.yaml")
    environment = make_vectorized_skill_environment(
        num_envs=config.num_envs,
        horizon=config.skill_horizon,
        scenario=config.skill_scenario,
        min_distance=config.skill_min_distance,
        max_distance=config.skill_max_distance,
        distractor_count=config.skill_distractor_count,
        randomization=scenario_randomization_from_config(config),
        num_skills=config.num_skills,
    )
    assert environment.num_skills == 16
    reset = environment.reset(jax.random.PRNGKey(0))
    step = jax.jit(environment.step)(
        jax.random.PRNGKey(1),
        reset.state,
        jnp.zeros((environment.num_envs,), dtype=jnp.int32),
    )
    assert step.reward.shape == (environment.num_envs,)
    full = load_skill_config("configs/skill_ppo_16_skill_randomized_26m.yaml")
    assert full.num_envs // full.num_skills == 16
    assert full.num_updates == 3200


@pytest.mark.parametrize(
    "path",
    sorted(Path("configs").glob("skill_ppo_*.yaml")),
    ids=lambda path: path.stem,
)
def test_skill_configs_put_every_skill_in_every_minibatch(path: Path) -> None:
    config = load_skill_config(path)
    assert config.num_envs % config.num_skills == 0
    assert (config.num_envs // config.num_skills) % config.num_minibatches == 0


def test_local_navigation_covers_the_new_targeted_skills() -> None:
    assert SkillId.EAT_COW in LOCAL_NAVIGATION_SKILLS
    for skill_id in (
        SkillId.CRAFT_STONE_SWORD,
        SkillId.COLLECT_COAL,
        SkillId.EAT_PLANT,
    ):
        assert skill_id in LOCAL_NAVIGATION_SKILLS
        result = evaluate_skill_policy(
            skill_id,
            policy="scripted",
            num_episodes=16,
            horizon=32,
            scenario=SkillScenario.LOCAL_NAVIGATION,
        )
        assert result["success_rate"] >= 0.75, (skill_id, result)
    result = evaluate_skill_policy(
        SkillId.EAT_COW,
        policy="scripted",
        num_episodes=16,
        horizon=32,
        scenario=SkillScenario.LOCAL_NAVIGATION,
    )
    # As in the nearby scenario, the cow can wander off the faced tile.
    assert result["success_rate"] >= 0.25, result


def test_planner_reads_the_new_fields() -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    assert set(PLANNER_FIELDS) >= {
        "survival.food",
        "survival.energy",
        "world.coal_reachable",
        "world.furnace_reachable",
        "world.ripe_plant_reachable",
        "world.grass_reachable",
        "world.cow_reachable",
    }
    assert read_field(state, "survival.food") == int(state.player_food)
    assert read_field(state, "survival.energy") == int(state.player_energy)
    assert read_field(state, "world.grass_reachable") == 1
    # The chain scenario scatters furnaces among its distractor blocks.
    assert read_field(state, "world.furnace_reachable") == int(
        np.any(np.asarray(state.map) == BlockType.FURNACE.value)
    )
    assert read_field(state, "world.cow_reachable") == int(bool(state.cows.mask.any()))


def test_planner_chains_the_new_skills_to_their_achievements() -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    for skill_id, length in (
        # wood x2, table, wood, pickaxe, stone, furnace
        (SkillId.PLACE_FURNACE, 7),
        # wood x2, table, wood, pickaxe, wood, stone, sword
        (SkillId.CRAFT_STONE_SWORD, 8),
        (SkillId.COLLECT_SAPLING, 1),
        (SkillId.PLACE_PLANT, 2),
    ):
        goal = (
            StateRequirement(
                achievement_field(SKILL_ACHIEVEMENTS[skill_id]), "at_least", 1
            ),
        )
        plan = shortest_plan(state, goal, ALL_SKILL_IDS)
        assert plan is not None, skill_id
        assert plan[1] == length, (skill_id, plan)
        # The stone-pickaxe catalog cannot reach any of them.
        assert shortest_plan(state, goal, STONE_PICKAXE_SKILL_IDS) is None

    tired = state.replace(player_energy=jnp.asarray(5, dtype=jnp.int32))
    plan = shortest_plan(
        tired,
        (StateRequirement("survival.energy", "at_least", 9),),
        ALL_SKILL_IDS,
    )
    assert plan == (frozenset({SkillId.SLEEP}), 1)


def test_open_ended_fields_grow_only_with_the_new_skills() -> None:
    old = open_ended_subgoal_fields(STONE_PICKAXE_SKILL_IDS)
    assert old[: len(SUBGOAL_FIELDS)] == SUBGOAL_FIELDS
    assert all(field.startswith("achievement.") for field in old[len(SUBGOAL_FIELDS) :])
    new = open_ended_subgoal_fields(ALL_SKILL_IDS)
    assert {"inventory.sapling", "survival.food", "survival.energy"} <= set(new)
    assert achievement_field(SKILL_ACHIEVEMENTS[SkillId.SLEEP]) in new
    assert VITAL_SUBGOAL_FIELDS["food"] == "survival.food"
    assert VITAL_SUBGOAL_FIELDS["energy"] == "survival.energy"


def test_memory_counts_grass_and_cows_as_obtainable() -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    # Cows keep spawning on grass, so none in sight does not rule them out.
    no_cows = state.replace(
        cows=state.cows.replace(mask=jnp.zeros_like(state.cows.mask))
    )
    memory = EpisodeMemory().observe(no_cows)
    assert memory.world_fact("world.grass_reachable") == 1
    assert memory.world_fact("world.cow_reachable") == 1
    assert memory.world_fact("world.coal_reachable") == 1


def test_nearby_scenarios_hold_what_each_new_skill_consumes() -> None:
    expected = {
        SkillId.CRAFT_STONE_SWORD: {"wood": 1, "stone": 1, "wood_pickaxe": 1},
        SkillId.COLLECT_COAL: {"wood_pickaxe": 1},
        SkillId.PLACE_STONE: {"stone": 1},
        SkillId.PLACE_FURNACE: {"stone": 1},
        SkillId.PLACE_PLANT: {"sapling": 1},
    }
    for skill_id, counts in expected.items():
        environment = make_skill_environment(skill_id, horizon=64)
        _, state = environment.reset(jax.random.PRNGKey(0))
        for item, count in counts.items():
            assert int(getattr(state.env_state.inventory, item)) >= count, (
                skill_id,
                item,
            )
    for skill_id in (SkillId.EAT_COW, SkillId.EAT_PLANT):
        _, state = make_skill_environment(skill_id, horizon=64).reset(
            jax.random.PRNGKey(0)
        )
        assert int(state.env_state.player_food) == 5
    _, state = make_skill_environment(SkillId.SLEEP, horizon=64).reset(
        jax.random.PRNGKey(0)
    )
    assert int(state.env_state.player_energy) == 8


@pytest.mark.parametrize("skill_id", NEW_SKILLS, ids=lambda skill: skill.value)
def test_chain_scripted_action_completes_the_new_skills(skill_id: SkillId) -> None:
    environment = make_skill_environment(
        skill_id,
        horizon=64,
        scenario=SkillScenario.RANDOMIZED,
        randomization=ScenarioRandomization(),
    )
    successes = 0
    for seed in range(4):
        _, state = environment.reset(jax.random.PRNGKey(seed))
        for step in range(64):
            action = _scripted_action(state.env_state, skill_id)
            result = environment.step(
                jax.random.PRNGKey(100 * seed + step),
                state,
                jnp.asarray(action, dtype=jnp.int32),
            )
            state = result.state
            if bool(result.done):
                break
        successes += bool(result.info["skill_success"])
    # A sapling drops one time in ten, so a run of misses is possible.
    assert successes >= (3 if skill_id is SkillId.COLLECT_SAPLING else 4)


def test_chain_switches_cover_every_skill() -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    skill_ids = jnp.arange(len(ALL_SKILL_IDS), dtype=jnp.int32)
    states = jax.tree_util.tree_map(
        lambda leaf: jnp.stack([leaf] * len(ALL_SKILL_IDS)), state
    )
    metrics = _batched_skill_metric(skill_ids, states)
    preconditions = _batched_skill_precondition(skill_ids, states)
    for index, skill_id in enumerate(ALL_SKILL_IDS):
        assert int(metrics[index]) == int(skill_metric(skill_id, state))
        assert bool(preconditions[index]) == bool(
            skill_precondition(SkillGoal(skill_id), state)
        )


def test_induction_offers_the_survival_fields_only_to_the_new_skills() -> None:
    assert induction_fields(STONE_PICKAXE_SKILL_IDS) == INDUCTION_FIELDS
    fields = induction_fields(ALL_SKILL_IDS)
    assert set(INDUCTION_FIELDS) < set(fields)
    assert {"survival.food", "survival.energy", "world.cow_reachable"} <= set(fields)
    assert not any(field.startswith("achievement.") for field in fields)
    assert {"survival.food", "survival.energy"} <= set(
        gain_fields(registry_model(ALL_SKILL_IDS))
    )
