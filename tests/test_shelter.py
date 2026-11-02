import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Achievement, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.game_logic import calculate_light_level

from tests.test_skill_interruption import ENVIRONMENT, RULE, _with_zombie, _without_mobs
from tests.test_subgoal_planner import (
    OBJECTIVE,
    ScriptedBackend,
    _answer,
    _select,
    _with_inventory,
)
from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.evaluation.chains import (
    ScriptedSkillExecutor,
    SkillExecutionResult,
    _execute_many,
    _run_shelter,
    evaluate_chain_episodes,
)
from tiny_sima.goals import ALL_SKILL_IDS, SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    SayCanPlanner,
    SequentialUtilityScorer,
    SkillOutcome,
)
from tiny_sima.planner.shelter import (
    Room,
    is_night,
    shelter_action,
    shelter_feasible,
)
from tiny_sima.planner.subgoals import SubgoalPlanner


def _at(state, timestep):
    return state.replace(
        timestep=jnp.asarray(timestep, dtype=state.timestep.dtype),
        light_level=calculate_light_level(timestep, EnvParams()),
    )


def _night(state, **inventory):
    """Early night, tired, with a wood pickaxe and ``inventory``."""

    state = _at(state, 160).replace(player_energy=jnp.asarray(5, dtype=jnp.int32))
    return _with_inventory(state, wood_pickaxe=1, **inventory)


def _sealed():
    """A night with the player in the top-left tile of a stone 2x2 room."""

    state = _night(_without_mobs(ENVIRONMENT.reset(jax.random.PRNGKey(0)).state))
    room = Room(tuple(int(value) for value in state.player_position))
    world_map = np.asarray(state.map).copy()
    for tile in room.tiles:
        world_map[tile] = BlockType.GRASS.value
    for tile, move in room.walls:
        step = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move]
        world_map[tile[0] + step[0], tile[1] + step[1]] = BlockType.STONE.value
    state = state.replace(map=jnp.asarray(world_map, dtype=state.map.dtype))
    return state, room


def test_every_wall_of_a_2x2_room_is_faced_from_inside() -> None:
    room = Room((5, 5))
    walls = room.walls
    assert len(walls) == len(set(walls)) == 8
    steps = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    for tile, move in walls:
        d_row, d_column = steps[move]
        # The wall is beyond the tile, and the tile behind it is in the room.
        assert (tile[0] + d_row, tile[1] + d_column) not in room.tiles
        assert (tile[0] - d_row, tile[1] - d_column) in room.tiles


def test_a_sealed_room_keeps_a_zombie_out_until_morning() -> None:
    state, room = _sealed()
    state = _with_zombie(state, (0, 3))

    result = _run_shelter(ENVIRONMENT, state, jax.random.PRNGKey(1))

    assert result.outcome is SkillOutcome.SUCCESS
    assert not is_night(result.state)
    assert int(result.state.player_health) == int(state.player_health)
    assert bool(result.state.achievements[Achievement.WAKE_UP.value])
    # The player mined one wall to get out.
    assert len(room.open_walls(np.asarray(result.state.map))) == 1


def test_the_shelter_is_built_and_slept_in() -> None:
    state = _night(
        _without_mobs(ENVIRONMENT.reset(jax.random.PRNGKey(0)).state), stone=8
    )
    result = _run_shelter(ENVIRONMENT, state, jax.random.PRNGKey(1))

    assert result.outcome is SkillOutcome.SUCCESS
    assert not is_night(result.state)
    assert int(result.state.inventory.stone) < 8
    assert bool(result.state.achievements[Achievement.PLACE_STONE.value])
    assert bool(result.state.achievements[Achievement.WAKE_UP.value])


def test_the_shelter_needs_night_food_and_a_pickaxe() -> None:
    state, _ = _sealed()
    assert shelter_feasible(state)
    assert not shelter_feasible(_at(state, 0))
    assert not shelter_feasible(_with_inventory(state, wood_pickaxe=0))
    assert not shelter_feasible(state.replace(player_drink=jnp.int32(2)))

    result = _run_shelter(ENVIRONMENT, _at(state, 0), jax.random.PRNGKey(1))
    assert result.outcome is SkillOutcome.INFEASIBLE
    assert result.steps == 0


def test_low_food_mines_a_way_out() -> None:
    state, _ = _sealed()
    hungry = state.replace(
        player_food=jnp.int32(2),
        player_direction=jnp.asarray(Action.LEFT.value, dtype=jnp.int32),
    )
    action, _ = shelter_action(hungry, None)
    assert action == Action.DO.value


def test_a_water_wall_is_drunk_from_instead_of_leaving() -> None:
    state, room = _sealed()
    world_map = np.asarray(state.map).copy()
    # The wall left of the player's tile.
    player = tuple(int(value) for value in state.player_position)
    world_map[player[0], player[1] - 1] = BlockType.WATER.value
    thirsty = state.replace(
        map=jnp.asarray(world_map, dtype=state.map.dtype),
        player_drink=jnp.int32(2),
        player_direction=jnp.asarray(Action.LEFT.value, dtype=jnp.int32),
    )
    action, _ = shelter_action(thirsty, room)
    assert action == Action.DO.value

    result = _run_shelter(ENVIRONMENT, thirsty, jax.random.PRNGKey(1))
    assert result.outcome is SkillOutcome.SUCCESS
    assert int(result.state.player_drink) > 2


def _night_planner(**options):
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    return SubgoalPlanner(backend, candidate_skills=(SkillId.COLLECT_WOOD,), **options)


def test_night_sends_the_planner_to_shelter() -> None:
    state, _ = _sealed()
    selection = _select(
        _night_planner(shelter_at_night=True), state, PlannerContext(OBJECTIVE)
    )
    assert selection.decision.skill_id is SkillId.SHELTER
    assert selection.plan_state.defending
    assert SkillId.SHELTER in {score.skill_id for score in selection.decision.scores}

    for planner, world in (
        (_night_planner(shelter_at_night=True), _at(state, 0)),
        (_night_planner(), state),
    ):
        selection = _select(planner, world, PlannerContext(OBJECTIVE))
        assert selection.decision.skill_id is SkillId.COLLECT_WOOD


class _Recording:
    def __init__(self):
        self.skill_ids = []

    def execute(self, environment, state, skill_id, rng, *, horizon, interruption):
        self.skill_ids.append(skill_id)
        return SkillExecutionResult(
            state=state,
            outcome=SkillOutcome.TIMEOUT,
            steps=1,
            skill_return=0.0,
            native_return=0.0,
        )


def test_the_shelter_never_reaches_the_executor() -> None:
    assert SkillId.SHELTER not in ALL_SKILL_IDS
    state, _ = _sealed()
    executor = _Recording()
    results = _execute_many(
        executor,
        ENVIRONMENT,
        [state, state],
        [SkillId.SHELTER, SkillId.COLLECT_WOOD],
        list(jax.random.split(jax.random.PRNGKey(0), 2)),
        horizon=8,
        interruption=RULE,
    )
    assert executor.skill_ids == [SkillId.COLLECT_WOOD]
    assert results[0].outcome is SkillOutcome.SUCCESS
    assert results[1].steps == 1


def test_the_trace_records_vitals_before_and_after() -> None:
    (episode,) = evaluate_chain_episodes(
        SayCanPlanner(utility_scorer=SequentialUtilityScorer((SkillId.COLLECT_WOOD,))),
        ScriptedSkillExecutor(),
        make_skill_chain_environment(),
        [jax.random.PRNGKey(0)],
        max_decisions=1,
    )
    step = episode.trace[0].to_dict()
    assert set(step["vitals_before"]) == {
        "health",
        "food",
        "drink",
        "energy",
        "sleeping",
        "daylight",
        "night",
    }
    assert step["vitals_before"]["health"] == 9
    assert step["vitals_after"]["food"] <= step["vitals_before"]["food"]
