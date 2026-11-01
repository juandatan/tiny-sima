import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import Action, BlockType

from tiny_sima.config import load_skill_config
from tiny_sima.envs.skills import (
    LOCAL_NAVIGATION_SKILLS,
    make_skill_environment,
    make_vectorized_skill_environment,
    scenario_randomization_from_config,
)
from tiny_sima.evaluation.chains import WOOD_TO_STONE_PICKAXE_CHAIN, chain_environment
from tiny_sima.goals import (
    ALL_SKILL_IDS,
    SKILL_ACHIEVEMENTS,
    SkillGoal,
    SkillId,
    StateRequirement,
    achievement_field,
    skill_metric,
    skill_mobs,
    skill_precondition,
    skill_success,
    skill_target_block,
)
from tiny_sima.memory import EpisodeMemory
from tiny_sima.planner.model_learning import EPISODE_START
from tiny_sima.planner.saycan import SkillOutcome
from tiny_sima.planner.symbolic import PLANNER_FIELDS, read_field, shortest_plan

COMBAT = {SkillId.DEFEAT_ZOMBIE: "zombie", SkillId.DEFEAT_SKELETON: "skeleton"}


def test_combat_skills_append_last() -> None:
    assert ALL_SKILL_IDS[-2:] == tuple(COMBAT)
    for skill_id in COMBAT:
        assert skill_target_block(skill_id) is None
        assert skill_id in LOCAL_NAVIGATION_SKILLS
        assert SKILL_ACHIEVEMENTS[skill_id].name == skill_id.name


@pytest.mark.parametrize("skill_id", tuple(COMBAT), ids=lambda skill: skill.value)
def test_scenario_places_one_live_mob_on_its_ground(skill_id: SkillId) -> None:
    environment = make_skill_environment(skill_id, horizon=96)
    _, state = environment.reset(jax.random.PRNGKey(0))
    mobs = skill_mobs(skill_id, state.env_state)
    target = np.asarray(state.target_position)
    ground = BlockType.PATH if skill_id is SkillId.DEFEAT_SKELETON else BlockType.GRASS

    assert int(np.count_nonzero(mobs.mask)) == 1
    np.testing.assert_array_equal(np.asarray(mobs.position[0]), target)
    assert int(state.env_state.map[target[0], target[1]]) == ground.value
    assert bool(skill_precondition(environment.goal, state.env_state))
    # The other mobs are gone for good, not just masked off.
    for kind in ("zombies", "cows", "skeletons"):
        others = getattr(state.env_state, kind)
        if others is not mobs:
            assert not np.any(np.asarray(others.health) > 0), kind


def test_a_kill_counts_and_a_despawn_does_not() -> None:
    environment = make_skill_environment(SkillId.DEFEAT_ZOMBIE, horizon=96)
    _, state = environment.reset(jax.random.PRNGKey(0))
    world = state.env_state
    zombies = world.zombies
    before = int(skill_metric(SkillId.DEFEAT_ZOMBIE, world))

    despawned = world.replace(
        zombies=zombies.replace(mask=jnp.zeros_like(zombies.mask))
    )
    assert int(skill_metric(SkillId.DEFEAT_ZOMBIE, despawned)) == before
    killed = world.replace(
        zombies=zombies.replace(
            mask=jnp.zeros_like(zombies.mask),
            health=zombies.health.at[0].set(0),
        )
    )
    assert int(skill_metric(SkillId.DEFEAT_ZOMBIE, killed)) == before + 1


def test_a_spawn_into_a_dead_slot_does_not_hide_the_first_kill() -> None:
    environment = make_skill_environment(SkillId.DEFEAT_ZOMBIE, horizon=96)
    _, initial = environment.reset(jax.random.PRNGKey(0))
    world = initial.env_state
    zombies = world.zombies
    achievement = SKILL_ACHIEVEMENTS[SkillId.DEFEAT_ZOMBIE].value
    # The live zombie dies, a new one spawns into a dead slot, and Craftax
    # unlocks the achievement: the dead slots alone are unchanged.
    killed_and_spawned = world.replace(
        zombies=zombies.replace(
            mask=zombies.mask.at[0].set(False).at[1].set(True),
            health=zombies.health.at[0].set(0).at[1].set(5),
        ),
        achievements=world.achievements.at[achievement].set(True),
    )
    assert bool(
        skill_success(SkillGoal(SkillId.DEFEAT_ZOMBIE), world, killed_and_spawned)
    )


def test_hitting_the_zombie_to_death_succeeds_and_unlocks_the_achievement() -> None:
    environment = make_skill_environment(SkillId.DEFEAT_ZOMBIE, horizon=96)
    _, state = environment.reset(jax.random.PRNGKey(0))
    # The nearby scenario faces a mob's tile, and the zombie stays adjacent.
    for step in range(16):
        result = environment.step(
            jax.random.PRNGKey(step), state, jnp.asarray(Action.DO.value)
        )
        state = result.state
        if bool(result.done):
            break
    assert bool(result.info["skill_success"])
    achievement = SKILL_ACHIEVEMENTS[SkillId.DEFEAT_ZOMBIE]
    assert bool(state.env_state.achievements[achievement.value])


def test_planner_reads_and_reaches_the_combat_achievements() -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    assert {"world.zombie_reachable", "world.skeleton_reachable"} <= set(PLANNER_FIELDS)
    assert dict(EPISODE_START)["world.zombie_reachable"] == 1
    # The chain world clears every mob.
    assert read_field(state, "world.zombie_reachable") == 0
    zombies = state.zombies.replace(mask=state.zombies.mask.at[0].set(True))
    with_zombie = state.replace(zombies=zombies)
    assert read_field(with_zombie, "world.zombie_reachable") == 1

    goal = (
        StateRequirement(
            achievement_field(SKILL_ACHIEVEMENTS[SkillId.DEFEAT_ZOMBIE]),
            "at_least",
            1,
        ),
    )
    assert shortest_plan(state, goal, ALL_SKILL_IDS) is None
    assert shortest_plan(with_zombie, goal, ALL_SKILL_IDS) == (
        frozenset({SkillId.DEFEAT_ZOMBIE}),
        1,
    )
    assert bool(skill_precondition(SkillGoal(SkillId.DEFEAT_ZOMBIE), with_zombie))


@pytest.mark.parametrize("skill_id", tuple(COMBAT), ids=lambda skill: skill.value)
def test_failed_hunts_rule_the_mob_out(skill_id: SkillId) -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    field = f"world.{COMBAT[skill_id]}_reachable"
    memory = EpisodeMemory().observe(state)
    # Mobs keep spawning, so none in sight does not rule them out.
    assert memory.world_fact(field) == 1
    for _ in range(2):
        memory = memory.record(state, skill_id, SkillOutcome.INFEASIBLE, 0, state)
    assert memory.ruled_out(COMBAT[skill_id])
    assert memory.world_fact(field) == 0


def test_eighteen_skill_config_builds_a_balanced_environment() -> None:
    config = load_skill_config("configs/skill_ppo_18_skill_randomized_debug.yaml")
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
    assert environment.num_skills == 18
    reset = environment.reset(jax.random.PRNGKey(0))
    step = jax.jit(environment.step)(
        jax.random.PRNGKey(1),
        reset.state,
        jnp.zeros((environment.num_envs,), dtype=jnp.int32),
    )
    assert step.reward.shape == (environment.num_envs,)
    full = load_skill_config("configs/skill_ppo_18_skill_randomized_29m.yaml")
    # The same samples per skill as the sixteen-skill run.
    assert full.num_envs // full.num_skills == 16
    assert full.num_updates == 3200
