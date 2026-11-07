import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import OBS_DIM, Action, BlockType

from tiny_sima.config import load_skill_config
from tiny_sima.envs.skills import (
    IRON_CRAFTING_SKILLS,
    LOCAL_NAVIGATION_SKILLS,
    RESOURCE_SKILLS,
    ScenarioRandomization,
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
    skill_precondition,
    skill_target_block,
)
from tiny_sima.memory import EpisodeMemory
from tiny_sima.planner.model_learning import EPISODE_START
from tiny_sima.planner.saycan import SkillOutcome
from tiny_sima.planner.symbolic import PLANNER_FIELDS, read_field, shortest_plan

IRON_STAGE = (
    SkillId.COLLECT_IRON,
    SkillId.CRAFT_IRON_PICKAXE,
    SkillId.CRAFT_IRON_SWORD,
    SkillId.COLLECT_DIAMOND,
)
CRAFTS = (SkillId.CRAFT_IRON_PICKAXE, SkillId.CRAFT_IRON_SWORD)
ORES = {SkillId.COLLECT_IRON: "iron", SkillId.COLLECT_DIAMOND: "diamond"}
SCENARIOS = ("nearby", "local_navigation", "randomized")


def _near(position: np.ndarray, block: np.ndarray) -> bool:
    """Return whether ``block`` is one of the eight tiles around ``position``."""

    return int(np.max(np.abs(block - position))) == 1


def test_iron_stage_appends_last_and_unlocks_the_last_achievements() -> None:
    assert len(ALL_SKILL_IDS) == 22
    assert ALL_SKILL_IDS[18:] == IRON_STAGE
    assert {SKILL_ACHIEVEMENTS[skill].name for skill in IRON_STAGE} == {
        "COLLECT_IRON",
        "MAKE_IRON_PICKAXE",
        "MAKE_IRON_SWORD",
        "COLLECT_DIAMOND",
    }
    # Every Craftax-Classic achievement now has a skill.
    assert len(set(SKILL_ACHIEVEMENTS.values())) == 22
    assert skill_target_block(SkillId.COLLECT_IRON) is BlockType.IRON
    assert skill_target_block(SkillId.COLLECT_DIAMOND) is BlockType.DIAMOND
    for skill_id in CRAFTS:
        assert skill_target_block(skill_id) is BlockType.CRAFTING_TABLE
    assert frozenset(CRAFTS) == IRON_CRAFTING_SKILLS
    assert set(IRON_STAGE) <= set(LOCAL_NAVIGATION_SKILLS)
    assert set(ORES) <= RESOURCE_SKILLS


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("skill_id", CRAFTS, ids=lambda skill: skill.value)
def test_the_approach_tile_is_near_both_the_table_and_a_furnace(
    skill_id: SkillId, scenario: str
) -> None:
    environment = make_skill_environment(
        skill_id,
        horizon=64,
        scenario=scenario,
        min_distance=1,
        max_distance=7,
        randomization=ScenarioRandomization() if scenario == "randomized" else None,
    )
    sides = set()
    for seed in range(16):
        _, state = environment.reset(jax.random.PRNGKey(seed))
        world = state.env_state
        world_map = np.asarray(world.map)
        table = np.asarray(state.target_position)
        approach = np.asarray(state.approach_position)
        player = np.asarray(world.player_position)
        furnaces = np.argwhere(world_map == BlockType.FURNACE.value)
        beside = [f for f in furnaces if int(np.abs(f - table).sum()) == 1]

        assert world_map[table[0], table[1]] == BlockType.CRAFTING_TABLE.value
        assert beside, seed
        assert any(_near(approach, furnace) for furnace in beside), seed
        assert _near(approach, table) or np.array_equal(approach, player)
        assert world_map[player[0], player[1]] != BlockType.FURNACE.value
        assert world_map[approach[0], approach[1]] != BlockType.FURNACE.value
        assert bool(skill_precondition(environment.goal, world))
        sides.update(tuple(furnace - table) for furnace in beside)
    # The furnace is not always on the same side.
    assert len(sides) > 1


@pytest.mark.parametrize("skill_id", CRAFTS, ids=lambda skill: skill.value)
def test_crafting_beside_the_table_and_furnace_makes_the_tool(skill_id) -> None:
    environment = make_skill_environment(skill_id, horizon=64)
    _, state = environment.reset(jax.random.PRNGKey(0))
    action = Action[skill_id.name.replace("CRAFT", "MAKE")]
    result = environment.step(jax.random.PRNGKey(1), state, jnp.asarray(action.value))

    assert bool(result.info["skill_success"])
    inventory = result.state.env_state.inventory
    assert int(inventory.iron) == 0 and int(inventory.coal) == 0
    achievement = SKILL_ACHIEVEMENTS[skill_id]
    assert bool(result.state.env_state.achievements[achievement.value])


@pytest.mark.parametrize(
    "skill_id",
    (SkillId.COLLECT_STONE, SkillId.CRAFT_WOOD_PICKAXE, SkillId.COLLECT_IRON),
    ids=lambda skill: skill.value,
)
def test_randomized_worlds_scatter_ore_around_other_skills(skill_id) -> None:
    environment = make_skill_environment(
        skill_id,
        horizon=64,
        scenario="randomized",
        min_distance=1,
        max_distance=7,
        randomization=ScenarioRandomization(),
    )
    seen = set()
    for seed in range(32):
        _, state = environment.reset(jax.random.PRNGKey(seed))
        world_map = np.asarray(state.env_state.map).copy()
        target = np.asarray(state.target_position)
        # The skill's own target is not a distractor.
        world_map[target[0], target[1]] = BlockType.GRASS.value
        row, column = np.asarray(state.env_state.player_position)
        rows, columns = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        other = world_map[
            row - rows : row + rows + 1, column - columns : column + columns + 1
        ]
        seen.update(
            ore for ore in (BlockType.IRON, BlockType.DIAMOND)
            if np.any(other == ore.value)
        )
    expected = {BlockType.IRON, BlockType.DIAMOND}
    # A skill's target is never among its own distractors.
    if skill_id is SkillId.COLLECT_IRON:
        expected = {BlockType.DIAMOND}
    assert seen >= expected


@pytest.mark.parametrize("skill_id", tuple(ORES), ids=lambda skill: skill.value)
def test_mining_the_ore_in_front_succeeds(skill_id: SkillId) -> None:
    environment = make_skill_environment(skill_id, horizon=96)
    _, state = environment.reset(jax.random.PRNGKey(0))
    # The nearby scenario faces away from the target: turn, then mine.
    for step, action in enumerate((state.target_direction, Action.DO.value)):
        result = environment.step(
            jax.random.PRNGKey(step), state, jnp.asarray(action)
        )
        state = result.state
    assert bool(result.info["skill_success"])


def test_diamond_needs_the_iron_pickaxe_and_iron_the_stone_one() -> None:
    for skill_id, tool in (
        (SkillId.COLLECT_DIAMOND, "iron_pickaxe"),
        (SkillId.COLLECT_IRON, "stone_pickaxe"),
    ):
        environment = make_skill_environment(skill_id, horizon=96)
        _, state = environment.reset(jax.random.PRNGKey(0))
        world = state.env_state
        without = world.replace(inventory=world.inventory.replace(**{tool: 0}))
        assert bool(skill_precondition(environment.goal, world))
        assert not bool(skill_precondition(environment.goal, without))


def test_planner_places_a_furnace_before_the_iron_pickaxe() -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    for field in ("world.iron_reachable", "world.diamond_reachable"):
        assert field in PLANNER_FIELDS
        assert dict(EPISODE_START)[field] == 1
    centre = np.asarray(state.player_position)
    # The chain world may hold a furnace among its distractors.
    world_map = jnp.where(
        state.map == BlockType.FURNACE.value, BlockType.GRASS.value, state.map
    )
    world_map = world_map.at[centre[0], centre[1] + 1].set(
        BlockType.CRAFTING_TABLE.value
    )
    holding = state.replace(
        map=world_map,
        inventory=state.inventory.replace(wood=1, stone=2, iron=1, coal=1),
    )
    assert read_field(holding, "world.furnace_reachable") == 0
    goal = (
        StateRequirement(
            achievement_field(SKILL_ACHIEVEMENTS[SkillId.CRAFT_IRON_PICKAXE]),
            "at_least",
            1,
        ),
    )
    assert shortest_plan(holding, goal, ALL_SKILL_IDS) == (
        frozenset({SkillId.PLACE_FURNACE}),
        2,
    )
    assert not bool(skill_precondition(SkillGoal(SkillId.CRAFT_IRON_PICKAXE), holding))


@pytest.mark.parametrize("skill_id", tuple(ORES), ids=lambda skill: skill.value)
def test_failed_mining_rules_the_ore_out(skill_id: SkillId) -> None:
    state = (
        chain_environment(WOOD_TO_STONE_PICKAXE_CHAIN)
        .reset(jax.random.PRNGKey(0))
        .state
    )
    field = f"world.{ORES[skill_id]}_reachable"
    memory = EpisodeMemory().observe(state)
    # Unseen ore is no evidence there is none.
    assert memory.world_fact(field) == 1
    for _ in range(2):
        memory = memory.record(state, skill_id, SkillOutcome.INFEASIBLE, 0, state)
    assert memory.ruled_out(ORES[skill_id])
    assert memory.world_fact(field) == 0


def test_twenty_two_skill_config_builds_a_balanced_environment() -> None:
    config = load_skill_config("configs/skill_ppo_22_skill_randomized_debug.yaml")
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
    assert environment.num_skills == 22
    reset = environment.reset(jax.random.PRNGKey(0))
    step = jax.jit(environment.step)(
        jax.random.PRNGKey(1),
        reset.state,
        jnp.zeros((environment.num_envs,), dtype=jnp.int32),
    )
    assert step.reward.shape == (environment.num_envs,)
    full = load_skill_config("configs/skill_ppo_22_skill_randomized_36m.yaml")
    # The same samples per skill and updates as the eighteen-skill run.
    assert full.num_envs // full.num_skills == 16
    assert full.num_updates == 3200
