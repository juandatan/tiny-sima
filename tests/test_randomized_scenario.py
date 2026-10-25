from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from craftax.craftax_classic.constants import OBS_DIM, BlockType

from tiny_sima.config import SkillPPOConfig, load_skill_config, save_config
from tiny_sima.envs.skills import (
    LOCAL_NAVIGATION_SKILLS,
    ScenarioRandomization,
    SkillScenario,
    SkillScenarioConfig,
    _route_mask,
    _scenario_inventory,
    make_skill_environment,
    make_vectorized_skill_environment,
    scenario_randomization_from_config,
    skills_for_scenario,
)
from tiny_sima.evaluation.chains import _scripted_action
from tiny_sima.evaluation.skills import scripted_skill_action
from tiny_sima.goals import SKILL_IDS, SkillId, skill_precondition

# Every knob at its local_navigation value: a cleared window, the player at
# the centre, one target, a reserved route and the training state.
_CONTROLLED = ScenarioRandomization(
    reserve_route=True,
    extra_target_count=0,
    table_probability=0.0,
    max_player_offset=0,
    clear_probability=1.0,
    random_inventory=False,
    random_vitals=False,
    max_start_timestep=0,
)
_TARGETS = {
    SkillId.COLLECT_WOOD: BlockType.TREE,
    SkillId.COLLECT_STONE: BlockType.STONE,
    SkillId.DRINK_WATER: BlockType.WATER,
    SkillId.CRAFT_WOOD_PICKAXE: BlockType.CRAFTING_TABLE,
    SkillId.CRAFT_WOOD_SWORD: BlockType.CRAFTING_TABLE,
    SkillId.CRAFT_STONE_PICKAXE: BlockType.CRAFTING_TABLE,
}


def _randomized(
    skill_id: SkillId,
    randomization: ScenarioRandomization | None = None,
    *,
    horizon: int = 64,
) -> Any:
    return make_skill_environment(
        skill_id,
        horizon=horizon,
        scenario=SkillScenario.RANDOMIZED,
        min_distance=1,
        max_distance=7,
        distractor_count=6,
        randomization=randomization,
    )


def _resets(environment: Any, seeds: range) -> list[Any]:
    reset = jax.jit(environment.reset)
    return [reset(jax.random.PRNGKey(seed)).state for seed in seeds]


def _view(state: Any) -> np.ndarray:
    row, column = (int(value) for value in np.asarray(state.player_position))
    rows, columns = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    return np.asarray(state.map)[
        row - rows : row + rows + 1,
        column - columns : column + columns + 1,
    ]


def _view_distances() -> np.ndarray:
    rows, columns = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    return (
        np.abs(np.arange(-rows, rows + 1))[:, None]
        + np.abs(np.arange(-columns, columns + 1))[None, :]
    )


@pytest.mark.parametrize("skill_id", SKILL_IDS)
def test_breadth_first_policy_solves_every_randomized_world(
    skill_id: SkillId,
) -> None:
    environment = _randomized(skill_id)
    reset = jax.jit(environment.reset)
    step = jax.jit(environment.step)

    for seed in range(6):
        state = reset(jax.random.PRNGKey(seed)).state
        assert bool(skill_precondition(environment.goal, state.env_state))
        succeeded = False
        for index in range(environment.goal.effective_horizon):
            action = _scripted_action(state.env_state, skill_id)
            result = step(jax.random.PRNGKey(index), state, jnp.asarray(action))
            state = result.state
            if bool(result.info["skill_success"]):
                succeeded = True
                break
            assert not bool(result.done)
        assert succeeded, f"seed {seed}"


@pytest.mark.parametrize("skill_id", LOCAL_NAVIGATION_SKILLS)
def test_randomized_targets_lie_in_view_within_the_distance_range(
    skill_id: SkillId,
) -> None:
    for state in _resets(_randomized(skill_id), range(8)):
        player = np.asarray(state.env_state.player_position)
        target = np.asarray(state.target_position)
        assert 1 <= int(np.abs(target - player).sum()) <= 7
        assert int(state.env_state.map[target[0], target[1]]) == (
            _TARGETS[skill_id].value
        )


def test_controlled_randomization_reproduces_the_local_navigation_setup() -> None:
    environment = _randomized(SkillId.COLLECT_STONE, _CONTROLLED)

    for state in _resets(environment, range(6)):
        env_state = state.env_state
        player = np.asarray(env_state.player_position)
        np.testing.assert_array_equal(player, np.asarray(env_state.map.shape) // 2)
        view = _view(env_state)
        # One target, six distractors and grass everywhere else.
        assert np.count_nonzero(view == BlockType.STONE.value) == 1
        assert np.count_nonzero(view != BlockType.GRASS.value) == 7
        assert not np.any(view == BlockType.CRAFTING_TABLE.value)
        route = np.asarray(
            _route_mask(
                env_state.map.shape,
                env_state.player_position,
                state.approach_position,
            )
        )
        assert np.all(np.asarray(env_state.map)[route] == BlockType.GRASS.value)
        expected = _scenario_inventory(SkillId.COLLECT_STONE)
        for name in ("wood", "stone", "sapling", "wood_pickaxe", "wood_sword"):
            assert int(getattr(env_state.inventory, name)) == int(
                getattr(expected, name)
            )
        assert int(env_state.player_health) == int(env_state.player_drink) == 9
        assert int(env_state.timestep) == 0
        assert float(env_state.light_level) == 1.0


def test_default_randomization_varies_position_holdings_and_clock() -> None:
    environment = _randomized(SkillId.COLLECT_WOOD)
    states = [state.env_state for state in _resets(environment, range(16))]
    centre = np.asarray(states[0].map.shape) // 2

    offsets = {
        tuple((np.asarray(state.player_position) - centre).tolist()) for state in states
    }
    assert len(offsets) > 1
    assert all(max(abs(value) for value in offset) <= 3 for offset in offsets)
    assert len({int(state.inventory.wood) for state in states}) > 1
    assert len({int(state.player_food) for state in states}) > 1
    timesteps = [int(state.timestep) for state in states]
    assert len(set(timesteps)) > 1
    assert all(0 <= timestep <= 300 for timestep in timesteps)
    assert min(float(state.light_level) for state in states) < 1.0
    for state in states:
        assert 3 <= int(state.player_health) <= 9


def test_extra_targets_are_no_closer_than_the_sampled_target() -> None:
    environment = _randomized(
        SkillId.COLLECT_WOOD,
        ScenarioRandomization(
            extra_target_count=4,
            max_player_offset=0,
            clear_probability=1.0,
        ),
    )
    distances = _view_distances()

    counts = []
    for state in _resets(environment, range(8)):
        player = np.asarray(state.env_state.player_position)
        target_distance = int(np.abs(np.asarray(state.target_position) - player).sum())
        trees = _view(state.env_state) == BlockType.TREE.value
        counts.append(int(trees.sum()))
        assert np.all(distances[trees] >= target_distance)
    assert max(counts) > 1
    assert all(count <= 5 for count in counts)


def test_table_probability_controls_tables_for_non_crafting_skills() -> None:
    def tables(probability: float) -> list[bool]:
        environment = _randomized(
            SkillId.DRINK_WATER,
            ScenarioRandomization(
                table_probability=probability,
                max_player_offset=0,
                clear_probability=1.0,
            ),
        )
        return [
            bool(np.any(_view(state.env_state) == BlockType.CRAFTING_TABLE.value))
            for state in _resets(environment, range(6))
        ]

    assert not any(tables(0.0))
    assert all(tables(1.0))


def test_drink_water_keeps_room_to_drink_under_random_vitals() -> None:
    environment = _randomized(SkillId.DRINK_WATER)

    for state in _resets(environment, range(16)):
        assert 3 <= int(state.env_state.player_drink) <= 8


def test_randomized_place_table_always_has_a_free_neighbour() -> None:
    environment = _randomized(SkillId.PLACE_TABLE)

    for state in _resets(environment, range(16)):
        assert bool(skill_precondition(environment.goal, state.env_state))
        assert int(state.env_state.inventory.wood) >= 2


def test_randomized_clock_extends_the_episode_limit() -> None:
    environment = _randomized(
        SkillId.COLLECT_WOOD,
        ScenarioRandomization(max_start_timestep=200),
        horizon=64,
    )
    controlled = _randomized(SkillId.COLLECT_WOOD, _CONTROLLED, horizon=64)

    assert environment.bundle.params.max_timesteps == 128 + 200
    assert controlled.bundle.params.max_timesteps == 128


def test_randomization_applies_only_to_the_randomized_kind() -> None:
    assert SkillScenarioConfig(SkillScenario.RANDOMIZED).randomization == (
        ScenarioRandomization()
    )
    assert SkillScenarioConfig().max_start_timestep == 0
    with pytest.raises(ValueError, match="randomized kind"):
        SkillScenarioConfig(
            SkillScenario.LOCAL_NAVIGATION,
            randomization=ScenarioRandomization(),
        )
    with pytest.raises(ValueError, match="extra_target_count"):
        ScenarioRandomization(extra_target_count=9)
    with pytest.raises(ValueError, match="table_probability"):
        ScenarioRandomization(table_probability=1.5)
    with pytest.raises(ValueError, match="max_player_offset"):
        ScenarioRandomization(max_player_offset=-1)
    with pytest.raises(ValueError, match="clear_probability"):
        ScenarioRandomization(clear_probability=-0.1)
    with pytest.raises(ValueError, match="max_start_timestep"):
        ScenarioRandomization(max_start_timestep=-1)


def test_vectorized_randomized_training_covers_place_table() -> None:
    randomization = ScenarioRandomization(extra_target_count=1)
    environment = make_vectorized_skill_environment(
        num_envs=12,
        horizon=64,
        scenario=SkillScenario.RANDOMIZED,
        min_distance=1,
        max_distance=7,
        randomization=randomization,
    )

    assert skills_for_scenario(SkillScenario.RANDOMIZED) == SKILL_IDS
    for skill_environment in environment.environments:
        assert skill_environment.scenario.kind is SkillScenario.RANDOMIZED
        assert skill_environment.scenario.randomization == randomization
    reset = environment.reset(jax.random.PRNGKey(0))
    assert reset.observation.shape[0] == 12


def test_straight_route_scripted_policy_rejects_randomized_worlds() -> None:
    environment = _randomized(SkillId.COLLECT_WOOD)
    state = environment.reset(jax.random.PRNGKey(0)).state

    with pytest.raises(ValueError, match="randomized"):
        scripted_skill_action(environment, state)


def test_skill_config_maps_randomization_knobs(tmp_path: Path) -> None:
    config = SkillPPOConfig(
        skill_horizon=64,
        skill_scenario="randomized",
        skill_min_distance=1,
        skill_max_distance=7,
        skill_reserve_route=True,
        skill_extra_target_count=3,
        skill_table_probability=0.25,
        skill_max_player_offset=2,
        skill_clear_probability=0.75,
        skill_random_inventory=False,
        skill_random_vitals=False,
        skill_max_start_timestep=150,
    ).validate()
    path = tmp_path / "randomized.yaml"
    save_config(config, path)

    assert load_skill_config(path) == config
    assert scenario_randomization_from_config(config) == ScenarioRandomization(
        reserve_route=True,
        extra_target_count=3,
        table_probability=0.25,
        max_player_offset=2,
        clear_probability=0.75,
        random_inventory=False,
        random_vitals=False,
        max_start_timestep=150,
    )
    assert (
        scenario_randomization_from_config(
            SkillPPOConfig(skill_scenario="local_navigation")
        )
        is None
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("skill_extra_target_count", 9),
        ("skill_table_probability", -0.5),
        ("skill_max_player_offset", 9),
        ("skill_clear_probability", 2.0),
        ("skill_max_start_timestep", -1),
    ],
)
def test_skill_config_rejects_invalid_randomization(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        SkillPPOConfig(skill_scenario="randomized", **{field: value}).validate()
