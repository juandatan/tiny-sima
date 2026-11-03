import json

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
from tiny_sima.planner.goal_selector import ScriptedGoalSelector
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
from tiny_sima.planner.subgoals import SubgoalPlanner, render_subgoal


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


def _night_planner(shelter: str = "now", **options):
    backend = ScriptedBackend(_answer(("inventory.wood", 1), shelter=shelter))
    options.setdefault("shelter_at_night", True)
    planner = SubgoalPlanner(
        backend, candidate_skills=(SkillId.COLLECT_WOOD,), **options
    )
    return planner, backend


def test_the_vlm_chooses_to_shelter_at_night() -> None:
    state, _ = _sealed()
    planner, backend = _night_planner()
    selection = _select(planner, state, PlannerContext(OBJECTIVE))

    assert selection.decision.skill_id is SkillId.SHELTER
    assert selection.plan_state.defending
    assert selection.plan_state.shelter == "now"
    assert SkillId.SHELTER in {score.skill_id for score in selection.decision.scores}
    # The sub-goals wait for the morning.
    assert [render_subgoal(goal) for goal in selection.plan_state.pending] == [
        "inventory.wood >= 1"
    ]
    (request,) = backend.requests
    assert "Also return shelter: now, after or no" in request.system_prompt
    assert (
        "Shelter: possible now. The nearest room has no water beside it. "
        "Morning is in 113 steps."
    ) in request.user_prompt


def test_the_planner_keeps_working_unless_the_vlm_shelters() -> None:
    state, _ = _sealed()
    for planner, world in (
        (_night_planner(shelter="no")[0], state),
        (_night_planner()[0], _at(state, 0)),
        (_night_planner(shelter_at_night=False)[0], state),
    ):
        selection = _select(planner, world, PlannerContext(OBJECTIVE))
        assert selection.decision.skill_id is SkillId.COLLECT_WOOD
        assert selection.plan_state.shelter == "no"
    # Without the option the prompt is unchanged.
    planner, backend = _night_planner(shelter_at_night=False)
    _select(planner, state, PlannerContext(OBJECTIVE))
    assert "shelter" not in backend.requests[0].system_prompt.lower()


def test_a_shelter_the_player_cannot_build_is_turned_down() -> None:
    state, _ = _sealed()
    planner, backend = _night_planner()
    selection = _select(
        planner, _with_inventory(state, wood_pickaxe=0), PlannerContext(OBJECTIVE)
    )
    assert selection.decision.skill_id is SkillId.COLLECT_WOOD
    assert "Shelter: not possible now: it needs a wood pickaxe. " in (
        backend.requests[0].user_prompt
    )
    assert any(
        event.subgoal == "shelter"
        and event.status == "rejected"
        and event.reason == "it needs a wood pickaxe"
        for event in selection.plan_state.events
    )


def test_the_shelter_lapses_at_dawn() -> None:
    state, _ = _sealed()
    planner, _ = _night_planner()
    night = _select(planner, state, PlannerContext(OBJECTIVE))
    context = PlannerContext(OBJECTIVE, plan_state=night.plan_state)

    again = _select(planner, state, context)
    assert again.decision.skill_id is SkillId.SHELTER
    morning = _select(planner, _at(state, 0), context)
    assert morning.decision.skill_id is SkillId.COLLECT_WOOD
    assert morning.plan_state.shelter == "no"


def test_the_vlm_can_drink_first_and_shelter_after() -> None:
    state, _ = _sealed()
    thirsty = state.replace(player_drink=jnp.int32(2))
    backend = ScriptedBackend(_answer(("survival.drink", 6), shelter="after"))
    planner = SubgoalPlanner(
        backend,
        candidate_skills=(SkillId.COLLECT_WOOD, SkillId.DRINK_WATER),
        shelter_at_night=True,
    )
    first = _select(planner, thirsty, PlannerContext(OBJECTIVE))
    # Low drink rules out sheltering now, but not drinking first.
    assert "Shelter: not possible now: drink is too low to stay in." in (
        backend.requests[0].user_prompt
    )
    assert first.decision.skill_id is SkillId.DRINK_WATER
    assert first.plan_state.shelter == "after"

    # Once drink is up, the shelter starts without asking again.
    context = PlannerContext(OBJECTIVE, plan_state=first.plan_state)
    then = _select(planner, state.replace(player_drink=jnp.int32(6)), context)
    assert then.decision.skill_id is SkillId.SHELTER
    assert then.plan_state.shelter == "now"
    assert ("shelter", "started") in {
        (event.subgoal, event.status) for event in then.plan_state.events
    }
    assert len(backend.requests) == 1

    # By day the choice lapses.
    morning = _select(planner, _at(thirsty, 0), context)
    assert morning.plan_state.shelter == "no"


def test_a_shelter_after_nothing_starts_now_and_alerts_still_ask() -> None:
    state, _ = _sealed()
    # 0 wood is always true, so nothing comes before the shelter.
    backend = ScriptedBackend(_answer(("inventory.wood", 0), shelter="after"))
    planner = SubgoalPlanner(
        backend,
        candidate_skills=(SkillId.COLLECT_WOOD,),
        shelter_at_night=True,
        vital_thresholds=(("health", 3),),
    )
    first = _select(planner, state, PlannerContext(OBJECTIVE))
    assert first.decision.skill_id is SkillId.SHELTER
    assert first.plan_state.shelter == "now"
    assert first.plan_state.pending == ()

    # The shelter alone asks nothing more, but low health still does.
    context = PlannerContext(OBJECTIVE, plan_state=first.plan_state)
    again = _select(planner, state, context)
    assert again.decision.skill_id is SkillId.SHELTER
    assert len(backend.requests) == 1
    _select(planner, state.replace(player_health=jnp.int32(3)), context)
    assert len(backend.requests) == 2


def test_the_curriculum_shelters_whenever_it_can_and_always_fights() -> None:
    state, _ = _sealed()
    selector = ScriptedGoalSelector(
        candidate_skills=(SkillId.COLLECT_WOOD,),
        shelter_at_night=True,
        choose_fights=True,
    )
    planner = SubgoalPlanner(
        selector,
        candidate_skills=(SkillId.COLLECT_WOOD,),
        shelter_at_night=True,
        choose_fights=True,
    )
    night = _select(planner, state, PlannerContext(OBJECTIVE))
    assert night.decision.skill_id is SkillId.SHELTER
    answer = json.loads(night.plan_state.raw_response)
    assert (answer["shelter"], answer["fight"]) == ("now", True)
    day = _select(planner, _at(state, 0), PlannerContext(OBJECTIVE))
    assert day.decision.skill_id is SkillId.COLLECT_WOOD
    assert json.loads(day.plan_state.raw_response)["shelter"] == "no"


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
