"""The goal source's game manual, vital rates, re-ask on a hit and rest."""

from dataclasses import replace

import jax
import jax.numpy as jnp

from tests.test_observation import _grass_world, _with_mobs
from tests.test_shelter import _at, _sealed
from tests.test_skill_interruption import ENVIRONMENT
from tests.test_subgoal_planner import (
    OBJECTIVE,
    ScriptedBackend,
    _answer,
    _attempt,
    _select,
)
from tests.test_vitals import RATES
from tiny_sima.evaluation.chains import _run_shelter
from tiny_sima.goals import SkillId
from tiny_sima.planner.saycan import PlannerContext, SkillOutcome
from tiny_sima.planner.subgoals import ReplanTrigger, SubgoalPlanner

FIGHT_SKILLS = (SkillId.COLLECT_WOOD, SkillId.DEFEAT_ZOMBIE)


def test_the_rates_are_told_and_projected() -> None:
    state = _grass_world(tree=[(0, 2)]).replace(
        player_health=jnp.int32(5),
        player_food=jnp.int32(4),
        player_drink=jnp.int32(2),
        player_energy=jnp.int32(9),
    )
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    planner = SubgoalPlanner(
        backend,
        candidate_skills=(SkillId.COLLECT_WOOD,),
        vital_rates=RATES,
    )
    _select(planner, state, PlannerContext(OBJECTIVE))

    (request,) = backend.requests
    assert "At these rates, steps until empty: food about 104" in (request.user_prompt)
    assert "Measured in this player's own staged play" in request.system_prompt

    plain = ScriptedBackend(_answer(("inventory.wood", 1)))
    _select(
        SubgoalPlanner(plain, candidate_skills=(SkillId.COLLECT_WOOD,)),
        state,
        PlannerContext(OBJECTIVE),
    )
    assert "At these rates" not in plain.requests[0].user_prompt
    assert "staged play" not in plain.requests[0].system_prompt


def test_the_manual_asks_the_plan_to_look_ahead() -> None:
    state = _grass_world(tree=[(0, 2)])
    for manual in (True, False):
        backend = ScriptedBackend(_answer(("inventory.wood", 1)))
        planner = SubgoalPlanner(
            backend, candidate_skills=(SkillId.COLLECT_WOOD,), manual=manual
        )
        _select(planner, state, PlannerContext(OBJECTIVE))
        system = backend.requests[0].system_prompt
        assert ("How the game works" in system) is manual
        assert ("planting a sapling early" in system) is manual
        assert ("Add an earlier sub-goal only when" in system) is not manual


def test_a_hit_in_a_fight_asks_again() -> None:
    state = _grass_world(tree=[(0, 2)])
    close = _with_mobs(state, "zombies", (1, 1))
    for replan_when_hurt, requests in ((True, 2), (False, 1)):
        backend = ScriptedBackend(_answer(("inventory.wood", 1), fight="now"))
        planner = SubgoalPlanner(
            backend,
            candidate_skills=FIGHT_SKILLS,
            threat_radius=2,
            defend_against=("zombie",),
            choose_fights=True,
            replan_when_hurt=replan_when_hurt,
        )
        first = _select(planner, close, PlannerContext(OBJECTIVE))
        assert first.decision is not None
        assert first.decision.skill_id is SkillId.DEFEAT_ZOMBIE
        assert first.plan_state.health == 9

        # The fight hands back at the hit.
        hit = close.replace(player_health=jnp.int32(7))
        attempts = (_attempt(SkillId.DEFEAT_ZOMBIE, SkillOutcome.INTERRUPTED),)
        context = PlannerContext(
            OBJECTIVE, history=attempts, plan_state=first.plan_state
        )
        again = _select(planner, hit, context)
        assert len(backend.requests) == requests
        if replan_when_hurt:
            assert again.plan_state.trigger is ReplanTrigger.HURT
            assert "The player was hit: health 9 to 7." in (
                backend.requests[1].user_prompt
            )
            assert len(again.plan_state.choices) == 2

    # Health lost with no threat close, such as to thirst, is not a hit.
    plan = replace(first.plan_state, health=9, fight="no")
    context = PlannerContext(
        OBJECTIVE, history=(_attempt(SkillId.COLLECT_WOOD),), plan_state=plan
    )
    planner = SubgoalPlanner(
        ScriptedBackend(_answer(("inventory.wood", 1))),
        candidate_skills=FIGHT_SKILLS,
        threat_radius=2,
        replan_when_hurt=True,
    )
    calm = _select(planner, state.replace(player_health=jnp.int32(7)), context)
    assert calm.plan_state.trigger is not ReplanTrigger.HURT

    # Nor is health lost with a zombie close while a need is empty.
    thirsty = close.replace(player_health=jnp.int32(7), player_drink=jnp.int32(0))
    plan = replace(first.plan_state, health=9, fight="no")
    context = PlannerContext(
        OBJECTIVE, history=(_attempt(SkillId.COLLECT_WOOD),), plan_state=plan
    )
    parched = _select(planner, thirsty, context)
    assert parched.plan_state.trigger is not ReplanTrigger.HURT


def _fight_planner(*answers: str, replan_when_hurt: bool = False):
    backend = ScriptedBackend(*answers)
    planner = SubgoalPlanner(
        backend,
        candidate_skills=FIGHT_SKILLS,
        threat_radius=2,
        defend_against=("zombie",),
        choose_fights=True,
        replan_when_hurt=replan_when_hurt,
    )
    return planner, backend


def test_a_fight_after_waits_for_the_first_subgoal() -> None:
    state = _grass_world(tree=[(0, 2)])
    close = _with_mobs(state, "zombies", (1, 1))
    planner, backend = _fight_planner(_answer(("inventory.wood", 1), fight="after"))
    first = _select(planner, close, PlannerContext(OBJECTIVE))
    assert first.decision is not None
    assert first.decision.skill_id is SkillId.COLLECT_WOOD
    assert first.plan_state.fight == "after"
    (choice,) = first.plan_state.choices
    assert (choice.kind, choice.answer) == ("fight", "after")
    assert "after reaches the first sub-goal first" in (
        backend.requests[0].system_prompt
    )

    # With the wood, the zombie still close is fought without asking again.
    wood = close.replace(inventory=close.inventory.replace(wood=1))
    context = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_WOOD),),
        plan_state=first.plan_state,
    )
    fought = _select(planner, wood, context)
    assert fought.decision is not None
    assert fought.decision.skill_id is SkillId.DEFEAT_ZOMBIE
    assert fought.plan_state.fight == "now"
    assert ("fight", "started") in {
        (event.subgoal, event.status) for event in fought.plan_state.events
    }
    assert len(backend.requests) == 1

    # Once the zombie goes, the fight for later lapses.
    gone = _select(planner, state, context)
    assert gone.plan_state.fight == "no"


def test_a_hit_asks_only_about_the_fight() -> None:
    state = _grass_world(tree=[(0, 2)])
    close = _with_mobs(state, "zombies", (1, 1))
    planner, backend = _fight_planner(
        _answer(("inventory.wood", 1), fight="no"),
        _answer(("inventory.sapling", 1), fight="now"),
        replan_when_hurt=True,
    )
    first = _select(planner, close, PlannerContext(OBJECTIVE))
    assert first.decision is not None
    assert first.decision.skill_id is SkillId.COLLECT_WOOD

    hit = close.replace(player_health=jnp.int32(7))
    context = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_WOOD, SkillOutcome.INTERRUPTED),),
        plan_state=first.plan_state,
    )
    again = _select(planner, hit, context)
    assert again.plan_state.trigger is ReplanTrigger.HURT
    assert "A hit asks only whether to fight and shelter" in (
        backend.requests[1].user_prompt
    )
    # The fight is taken, and the sub-goals stand.
    assert again.decision is not None
    assert again.decision.skill_id is SkillId.DEFEAT_ZOMBIE
    assert again.plan_state.fight == "now"
    assert again.plan_state.pending == first.plan_state.pending


def _rest_planner():
    backend = ScriptedBackend(_answer(("inventory.wood", 1), shelter="now"))
    planner = SubgoalPlanner(
        backend,
        candidate_skills=(SkillId.COLLECT_WOOD,),
        shelter_at_night=True,
        rest_by_day=True,
    )
    return planner, backend


def test_the_shelter_is_offered_by_day_to_rest() -> None:
    state, _ = _sealed()
    day = _at(state, 0).replace(player_health=jnp.int32(5))
    planner, backend = _rest_planner()
    selection = _select(planner, day, PlannerContext(OBJECTIVE))

    assert selection.decision is not None
    assert selection.decision.skill_id is SkillId.SHELTER
    assert selection.plan_state.shelter == "now"
    (choice,) = selection.plan_state.choices
    assert (choice.kind, choice.answer, choice.possible) == ("rest", "now", True)
    (request,) = backend.requests
    assert "By day, while health is not full, the same choice rests" in (
        request.system_prompt
    )
    assert (
        "Resting in a shelter: possible now. The nearest room has no water beside it."
    ) in request.user_prompt
    assert "Morning is in" not in request.user_prompt

    # A rest that ended well is not taken again unasked.
    context = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.SHELTER),),
        plan_state=selection.plan_state,
    )
    rested = _select(planner, day.replace(player_health=jnp.int32(8)), context)
    assert rested.plan_state.shelter == "no"
    assert ("shelter", "ended", "done") in {
        (event.subgoal, event.status, event.reason)
        for event in rested.plan_state.events
    }

    # With full health, or without the option, nothing is offered by day.
    for planner, world in (
        (_rest_planner()[0], _at(state, 0)),
        (_night_only_planner(), day),
    ):
        selection = _select(planner, world, PlannerContext(OBJECTIVE))
        assert selection.decision is not None
        assert selection.decision.skill_id is SkillId.COLLECT_WOOD
        assert selection.plan_state.choices == ()


def _night_only_planner():
    return SubgoalPlanner(
        ScriptedBackend(_answer(("inventory.wood", 1), shelter="now")),
        candidate_skills=(SkillId.COLLECT_WOOD,),
        shelter_at_night=True,
    )


def test_a_day_rest_stays_in_until_health_is_full() -> None:
    state, _ = _sealed()
    day = _at(state, 0).replace(player_health=jnp.int32(6))

    result = _run_shelter(ENVIRONMENT, day, jax.random.PRNGKey(1))

    assert result.outcome is SkillOutcome.SUCCESS
    assert int(result.state.player_health) == 9
    assert result.steps > 0
