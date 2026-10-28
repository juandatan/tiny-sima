from collections.abc import Sequence

import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.game_logic import calculate_light_level

from tests.test_observation import _grass_world, _with_mobs
from tests.test_subgoal_planner import (
    OBJECTIVE,
    ScriptedBackend,
    _answer,
    _attempt,
    _select,
    _state,
    _with_inventory,
)
from tiny_sima.goals import SkillId
from tiny_sima.memory import EpisodeMemory
from tiny_sima.planner import (
    OPEN_ENDED_SUBGOAL_FIELDS,
    TEXT_ONLY,
    SubgoalRequest,
    build_subgoal_system_prompt,
)
from tiny_sima.planner.saycan import PlannerContext, SkillOutcome
from tiny_sima.planner.subgoals import ReplanTrigger, SubgoalPlanner


def _follow(planner, state, plan, attempts):
    context = PlannerContext(OBJECTIVE, history=tuple(attempts), plan_state=plan)
    return _select(planner, state, context).plan_state


def test_a_subgoal_times_out_after_its_plan_length_plus_slack() -> None:
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    planner = SubgoalPlanner(backend, subgoal_slack=1)
    state = _state()

    plan = _select(planner, state, PlannerContext(OBJECTIVE)).plan_state
    assert plan.current_budget == 2
    assert plan.skills_on_current == 0

    # Successes that leave the inventory unchanged spend the budget.
    attempts = [_attempt(SkillId.COLLECT_WOOD)]
    plan = _follow(planner, state, plan, attempts)
    assert plan.trigger is None
    assert plan.skills_on_current == 1
    attempts.append(_attempt(SkillId.COLLECT_WOOD))
    plan = _follow(planner, state, plan, attempts)
    assert plan.trigger is ReplanTrigger.TIMED_OUT
    assert [event.status for event in plan.events][-2:] == ["timed_out", "proposed"]
    assert "not reached in 2 skills (budget 2)" in plan.events[-2].reason
    assert "within its skill budget: inventory.wood >= 1 after 2 skills." in (
        backend.requests[-1].user_prompt
    )
    # The re-proposed sub-goal starts a fresh budget.
    assert plan.skills_on_current == 0
    assert plan.current_budget == 2


def test_no_slack_never_times_out() -> None:
    planner = SubgoalPlanner(
        ScriptedBackend(_answer(("inventory.wood", 1))), subgoal_slack=None
    )
    state = _state()
    plan = _select(planner, state, PlannerContext(OBJECTIVE)).plan_state
    assert plan.current_budget is None
    attempts = [_attempt(SkillId.COLLECT_WOOD)] * 6
    assert _follow(planner, state, plan, attempts).trigger is None


def test_a_failed_skill_reports_the_memory_cause() -> None:
    backend = ScriptedBackend(_answer(("inventory.stone", 1)))
    planner = SubgoalPlanner(backend)
    # No stone anywhere in view or in memory.
    state = _with_inventory(_grass_world(tree=[(1, 1)]), wood_pickaxe=1)
    memory = EpisodeMemory().observe(state)
    plan = _select(planner, state, PlannerContext(OBJECTIVE, memory=memory)).plan_state

    memory = memory.record(
        state, SkillId.COLLECT_STONE, SkillOutcome.TIMEOUT, 96, state
    )
    context = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_STONE, SkillOutcome.TIMEOUT),),
        memory=memory,
        plan_state=plan,
    )
    failed = _select(planner, state, context).plan_state
    assert failed.trigger is ReplanTrigger.SKILL_FAILED
    assert failed.trigger_detail == "collect_stone timeout, target not seen"
    skill_failed = [event for event in failed.events if event.status == "skill_failed"]
    assert skill_failed[0].reason == "collect_stone timeout, target not seen"
    prompt = backend.requests[-1].user_prompt
    assert (
        "Why you are asked now: The last skill failed: collect_stone timeout, "
        "target not seen." in prompt
    )
    assert "inventory.stone >= 1: skill_failed" in prompt


def test_several_low_vitals_raise_one_alert() -> None:
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    planner = SubgoalPlanner(
        backend, vital_thresholds=(("food", 3), ("drink", 3), ("energy", 3))
    )
    state = _state()
    plan = _select(planner, state, PlannerContext(OBJECTIVE)).plan_state

    hungry = state.replace(player_food=2, player_drink=1)
    raised = _follow(planner, hungry, plan, [_attempt(SkillId.COLLECT_WOOD)])
    assert raised.trigger is ReplanTrigger.LOW_VITALS
    assert raised.alerts == ("food", "drink")
    assert "Food and drink are low." in backend.requests[-1].user_prompt

    # Food recovers and energy drops: only energy is new.
    tired = state.replace(player_drink=1, player_energy=3)
    again = _follow(planner, tired, raised, [_attempt(SkillId.COLLECT_WOOD)] * 2)
    assert again.trigger is ReplanTrigger.LOW_VITALS
    assert again.trigger_detail == "energy"
    assert again.alerts == ("drink", "energy")


def test_a_close_zombie_raises_one_threat_alert() -> None:
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    planner = SubgoalPlanner(backend, threat_radius=2)
    state = _grass_world(tree=[(0, 2)])
    plan = _select(planner, state, PlannerContext(OBJECTIVE)).plan_state

    # A cow or a distant zombie is no threat.
    calm = _with_mobs(_with_mobs(state, "cows", (1, 0)), "zombies", (3, 3))
    plan = _follow(planner, calm, plan, [_attempt(SkillId.COLLECT_WOOD)])
    assert plan.trigger is None

    close = _with_mobs(state, "zombies", (1, 1))
    attempts = [_attempt(SkillId.COLLECT_WOOD)] * 2
    plan = _follow(planner, close, plan, attempts)
    assert plan.trigger is ReplanTrigger.THREAT
    assert plan.trigger_detail == "zombie at offset [1, 1]"
    assert len(backend.requests) == 2

    attempts.append(_attempt(SkillId.COLLECT_WOOD))
    plan = _follow(planner, close, plan, attempts)
    assert plan.trigger is None
    attempts.append(_attempt(SkillId.COLLECT_WOOD))
    plan = _follow(planner, state, plan, attempts)
    assert plan.alerts == ()


def test_nightfall_asks_once() -> None:
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    planner = SubgoalPlanner(backend, replan_at_night=True)
    params = EnvParams()
    day = _state().replace(
        timestep=jnp.int32(0), light_level=calculate_light_level(0, params)
    )
    night = day.replace(
        timestep=jnp.int32(210), light_level=calculate_light_level(210, params)
    )
    plan = _select(planner, day, PlannerContext(OBJECTIVE)).plan_state

    plan = _follow(planner, night, plan, [_attempt(SkillId.COLLECT_WOOD)])
    assert plan.trigger is ReplanTrigger.NIGHT
    assert "Night has fallen." in backend.requests[-1].user_prompt
    plan = _follow(planner, night, plan, [_attempt(SkillId.COLLECT_WOOD)] * 2)
    assert plan.trigger is None


def test_text_only_observation_sends_no_image() -> None:
    backend = ScriptedBackend(_answer(("inventory.wood", 1)))
    _select(
        SubgoalPlanner(backend, observation=TEXT_ONLY),
        _state(),
        PlannerContext(OBJECTIVE),
    )
    request = backend.requests[0]
    assert request.image is None
    assert "Text observation: " in request.user_prompt
    assert request.user_prompt.endswith("The local map shows the agent's surroundings.")
    assert "local_map" in request.user_prompt
    assert request.system_prompt.startswith(build_subgoal_system_prompt())

    default = ScriptedBackend(_answer(("inventory.wood", 1)))
    _select(SubgoalPlanner(default), _state(), PlannerContext(OBJECTIVE))
    assert default.requests[0].image is not None
    assert "Text observation" not in default.requests[0].user_prompt
    assert default.requests[0].system_prompt == build_subgoal_system_prompt()


def test_achievement_subgoals_are_planned() -> None:
    assert "achievement" not in build_subgoal_system_prompt()
    backend = ScriptedBackend(_answer(("achievement.make_wood_pickaxe", 1)))
    planner = SubgoalPlanner(backend, fields=OPEN_ENDED_SUBGOAL_FIELDS)
    selection = _select(planner, _state(), PlannerContext(OBJECTIVE))
    assert selection.decision is not None
    assert selection.decision.skill_id is SkillId.COLLECT_WOOD
    assert "An achievement field is 1 once" in backend.requests[0].system_prompt


class FixedSource:
    """A goal source that answers without the VLM."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.seen: list[SubgoalRequest] = []

    def propose_subgoals(self, requests: Sequence[SubgoalRequest]) -> list[str]:
        self.seen.extend(requests)
        return [self.answer] * len(requests)


def test_a_subgoal_source_replaces_the_vlm() -> None:
    source = FixedSource(_answer(("inventory.iron_pickaxe", 1)))
    planner = SubgoalPlanner(source, max_rejections=1)
    selection = _select(planner, _state(), PlannerContext(OBJECTIVE))
    # Its answers are checked like the VLM's: both rounds are rejected.
    assert [request.trigger for request in source.seen] == [ReplanTrigger.START] * 2
    assert source.seen[1].feedback
    assert selection.plan_state.rejections == 2
    assert selection.decision is None
