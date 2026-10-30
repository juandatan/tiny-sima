import json
from collections.abc import Sequence

import jax
import pytest

from tiny_sima.envs.chains import make_skill_chain_environment
from tiny_sima.goals import SkillId
from tiny_sima.planner.saycan import (
    PlannerContext,
    PlannerSelectionStatus,
    SkillAttempt,
    SkillOutcome,
)
from tiny_sima.planner.subgoals import (
    SUBGOAL_FIELDS,
    ReplanTrigger,
    SubgoalPlan,
    SubgoalPlanner,
    parse_subgoal_response,
    subgoal_output_type,
)
from tiny_sima.planner.vlm import VLMRequest

OBJECTIVE = "Get a stone pickaxe."


def _state(seed: int = 0):
    return make_skill_chain_environment().reset(jax.random.PRNGKey(seed)).state


def _with_inventory(state, **counts):
    return state.replace(inventory=state.inventory.replace(**counts))


def _answer(*subgoals: tuple[str, int], assessment: str = "") -> str:
    return json.dumps(
        {
            "assessment": assessment,
            "subgoals": [
                {"field": field, "at_least": value} for field, value in subgoals
            ],
        }
    )


class ScriptedBackend:
    """Return queued answers in order, repeating the last one."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.requests: list[VLMRequest] = []
        self.batches: list[int] = []

    def generate(self, request: VLMRequest) -> str:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[VLMRequest]) -> list[str]:
        self.batches.append(len(requests))
        responses = []
        for request in requests:
            self.requests.append(request)
            responses.append(
                self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            )
        return responses


def _attempt(skill_id: SkillId, outcome: SkillOutcome = SkillOutcome.SUCCESS):
    return SkillAttempt(skill_id, outcome, steps=8, native_return=0.0)


def _select(planner: SubgoalPlanner, state, context: PlannerContext):
    return planner.select_many([state], [context])[0]


def test_first_decision_asks_for_subgoals_and_follows_the_plan() -> None:
    backend = ScriptedBackend(_answer(("inventory.stone_pickaxe", 1)))
    planner = SubgoalPlanner(backend)
    state = _state()

    selection = _select(planner, state, PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.SELECTED
    assert selection.decision is not None
    assert selection.decision.skill_id is SkillId.COLLECT_WOOD
    plan = selection.plan_state
    assert isinstance(plan, SubgoalPlan)
    assert plan.trigger is ReplanTrigger.START
    assert plan.vlm_calls == 1
    assert [event.status for event in plan.events] == ["proposed"]
    request = backend.requests[0]
    assert request.output_type is subgoal_output_type(SUBGOAL_FIELDS, 4)
    assert "Objective: Get a stone pickaxe." in request.user_prompt
    assert "Known game mechanics" in request.system_prompt

    # The next decision follows the plan without asking again.
    later = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_WOOD),),
        plan_state=plan,
    )
    following = _select(planner, _with_inventory(state, wood=1), later)
    assert len(backend.requests) == 1
    assert following.decision is not None
    assert following.decision.skill_id is SkillId.COLLECT_WOOD
    assert following.plan_state.trigger is None


def test_rejected_subgoals_go_back_with_the_reason() -> None:
    backend = ScriptedBackend(
        _answer(("inventory.iron_pickaxe", 1)),
        _answer(("inventory.stone_pickaxe", 1)),
    )
    planner = SubgoalPlanner(backend)

    selection = _select(planner, _state(), PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.SELECTED
    assert len(backend.requests) == 2
    retry = backend.requests[1].user_prompt
    assert "Planner rejected your last answer" in retry
    assert "inventory.iron_pickaxe >= 1" in retry
    plan = selection.plan_state
    assert plan.rejections == 1
    assert plan.vlm_calls == 2
    assert [event.status for event in plan.events] == ["rejected", "proposed"]


def test_every_answer_rejected_stops_with_the_reason() -> None:
    backend = ScriptedBackend(_answer(("inventory.iron_sword", 1)))
    planner = SubgoalPlanner(backend, max_rejections=2)

    selection = _select(planner, _state(), PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.NO_VIABLE_SKILL
    assert selection.decision is None
    assert selection.utility_error is not None
    assert "inventory.iron_sword" in selection.utility_error
    assert len(backend.requests) == 3
    assert selection.plan_state.rejections == 3


def test_subgoals_that_already_hold_go_back_with_the_current_value() -> None:
    backend = ScriptedBackend(
        _answer(("inventory.wood_pickaxe", 1)),
        _answer(("inventory.stone_pickaxe", 1)),
    )
    planner = SubgoalPlanner(backend)
    state = _with_inventory(_state(), wood_pickaxe=1)

    selection = _select(planner, state, PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.SELECTED
    retry = backend.requests[1].user_prompt
    assert (
        "inventory.wood_pickaxe >= 1 already holds "
        "(inventory.wood_pickaxe is 1)." in retry
    )
    assert "Give sub-goals that are not yet true" in retry
    assert "do not propose it again" not in retry
    plan = selection.plan_state
    assert plan.rejections == 1
    held = plan.events[0]
    assert held.status == "already_held"
    assert held.reason == "inventory.wood_pickaxe is 1"


def test_a_held_subgoal_proposed_again_is_called_a_repeat() -> None:
    backend = ScriptedBackend(
        _answer(("survival.drink", 4)),
        _answer(("survival.drink", 4)),
        _answer(("inventory.stone_pickaxe", 1)),
    )
    planner = SubgoalPlanner(backend)
    state = _state()
    state = state.replace(player_drink=4)

    selection = _select(planner, state, PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.SELECTED
    first, second = (request.user_prompt for request in backend.requests[1:])
    assert "survival.drink >= 4 already holds (survival.drink is 4)." in first
    assert (
        "survival.drink >= 4 already holds (survival.drink is 4) and was "
        "reported before; do not propose it again." in second
    )
    assert selection.plan_state.rejections == 2


def test_a_zero_count_subgoal_is_rejected_as_always_true() -> None:
    backend = ScriptedBackend(
        _answer(("inventory.stone_pickaxe", 0)),
        _answer(("inventory.stone_pickaxe", 1)),
    )
    planner = SubgoalPlanner(backend)

    selection = _select(planner, _state(), PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.SELECTED
    retry = backend.requests[1].user_prompt
    assert (
        "inventory.stone_pickaxe >= 0: always true; at_least is the count to "
        "reach, at least 1" in retry
    )
    assert "already holds" not in retry
    plan = selection.plan_state
    assert plan.rejections == 1
    assert [event.status for event in plan.events] == ["rejected", "proposed"]


def test_only_subgoals_that_already_hold_stop_with_the_value() -> None:
    backend = ScriptedBackend(_answer(("inventory.wood_pickaxe", 1)))
    planner = SubgoalPlanner(backend, max_rejections=1)
    state = _with_inventory(_state(), wood_pickaxe=1)

    selection = _select(planner, state, PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.NO_VIABLE_SKILL
    assert selection.utility_error == (
        "no reachable sub-goal: inventory.wood_pickaxe >= 1: "
        "inventory.wood_pickaxe is 1"
    )


def test_malformed_answers_are_retried() -> None:
    backend = ScriptedBackend("not json", _answer(("inventory.stone_pickaxe", 1)))
    selection = _select(SubgoalPlanner(backend), _state(), PlannerContext(OBJECTIVE))

    assert selection.status is PlannerSelectionStatus.SELECTED
    assert selection.plan_state.invalid_responses == 1
    assert "malformed" in backend.requests[1].user_prompt


def test_reached_subgoals_advance_and_asking_resumes_when_none_remain() -> None:
    backend = ScriptedBackend(
        _answer(("inventory.wood", 1), ("inventory.wood_pickaxe", 1)),
        _answer(("inventory.stone_pickaxe", 1)),
    )
    planner = SubgoalPlanner(backend)
    state = _state()
    plan = _select(planner, state, PlannerContext(OBJECTIVE)).plan_state

    holding = _with_inventory(state, wood=1)
    context = PlannerContext(
        OBJECTIVE, history=(_attempt(SkillId.COLLECT_WOOD),), plan_state=plan
    )
    advanced = _select(planner, holding, context).plan_state
    assert len(backend.requests) == 1
    assert len(advanced.pending) == 1
    assert advanced.to_dict()["events"] == [
        {
            "decision_index": 1,
            "subgoal": "inventory.wood >= 1",
            "status": "achieved",
            "reason": "",
        }
    ]

    done = _with_inventory(state, wood_pickaxe=1)
    context = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_WOOD),) * 4,
        plan_state=advanced,
    )
    replanned = _select(planner, done, context)
    assert len(backend.requests) == 2
    assert replanned.plan_state.trigger is ReplanTrigger.SUBGOALS_DONE
    assert "Every sub-goal so far has been reached" in backend.requests[1].user_prompt


def test_a_failed_skill_and_low_drink_each_trigger_one_call() -> None:
    backend = ScriptedBackend(_answer(("inventory.stone_pickaxe", 1)))
    planner = SubgoalPlanner(backend)
    state = _state()
    plan = _select(planner, state, PlannerContext(OBJECTIVE)).plan_state

    failed = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_WOOD, SkillOutcome.TIMEOUT),),
        plan_state=plan,
    )
    after_failure = _select(planner, state, failed).plan_state
    assert after_failure.trigger is ReplanTrigger.SKILL_FAILED
    assert len(backend.requests) == 2

    thirsty = state.replace(player_drink=2)
    context = PlannerContext(
        OBJECTIVE,
        history=(_attempt(SkillId.COLLECT_WOOD),) * 2,
        plan_state=after_failure,
    )
    raised = _select(planner, thirsty, context).plan_state
    assert raised.trigger is ReplanTrigger.LOW_VITALS
    assert raised.alerts == ("drink",)
    assert len(backend.requests) == 3
    assert "Why you are asked now: Drink is low." in backend.requests[-1].user_prompt
    # Still thirsty at the next decision, but already asked.
    context = PlannerContext(
        OBJECTIVE, history=(_attempt(SkillId.COLLECT_WOOD),) * 3, plan_state=raised
    )
    assert _select(planner, thirsty, context).plan_state.trigger is None
    assert len(backend.requests) == 3


class TreelessMemory:
    """A memory that believes every tree is gone."""

    def observe(self, state):
        return self

    def record(self, pre_state, skill_id, outcome, steps, post_state):
        return self

    def world_fact(self, field: str) -> int | None:
        return 0 if field == "world.tree_reachable" else None

    def summary(self, state=None):
        return {"resources": {"tree": {"ruled_out": True}}}


def test_memory_belief_replaces_the_map_scan() -> None:
    context = PlannerContext(OBJECTIVE, memory=TreelessMemory())
    answer = _answer(("inventory.stone_pickaxe", 1))

    believed = _select(SubgoalPlanner(ScriptedBackend(answer)), _state(), context)
    assert believed.status is PlannerSelectionStatus.NO_VIABLE_SKILL
    assert believed.utility_error is not None
    assert "is false" in believed.utility_error

    backend = ScriptedBackend(answer)
    scanned = _select(
        SubgoalPlanner(backend, use_memory_belief=False), _state(), context
    )
    assert scanned.status is PlannerSelectionStatus.SELECTED
    # The memory summary still reaches the prompt.
    assert '"ruled_out": true' in backend.requests[0].user_prompt


def test_episodes_needing_the_vlm_share_one_batch() -> None:
    backend = ScriptedBackend(_answer(("inventory.stone_pickaxe", 1)))
    planner = SubgoalPlanner(backend)
    states = [_state(0), _state(1), _state(2)]
    first = planner.select_many(states, [PlannerContext(OBJECTIVE)] * 3)

    assert backend.batches == [3]
    contexts = [
        PlannerContext(OBJECTIVE, plan_state=first[0].plan_state),
        PlannerContext(
            OBJECTIVE,
            history=(_attempt(SkillId.COLLECT_WOOD),),
            plan_state=first[1].plan_state,
        ),
        PlannerContext(
            OBJECTIVE,
            history=(_attempt(SkillId.COLLECT_WOOD, SkillOutcome.TIMEOUT),),
            plan_state=first[2].plan_state,
        ),
    ]
    planner.select_many(states, contexts)
    # Episode 0 is back at decision 0 with a plan: no call. Episode 1 follows
    # its plan. Only episode 2's failure asks.
    assert backend.batches[-1] == 1
    assert all(json.dumps(selection.plan_state.to_dict()) for selection in first)


def test_parse_checks_shape_only() -> None:
    parsed = parse_subgoal_response(_answer(("inventory.gold", 2), assessment="hi"))
    assert parsed.valid
    assert parsed.assessment == "hi"
    assert parsed.subgoals[0][0].field == "inventory.gold"
    assert not parse_subgoal_response('{"subgoals": []}').valid
    assert not parse_subgoal_response(
        '{"subgoals": [{"field": "inventory.wood", "at_least": "two"}]}'
    ).valid
    trimmed = parse_subgoal_response(
        _answer(*[("inventory.wood", value) for value in range(1, 7)]), max_subgoals=4
    )
    assert len(trimmed.subgoals) == 4


def test_output_schema_lists_the_observable_fields() -> None:
    schema = subgoal_output_type().model_json_schema()
    assert list(schema["properties"]) == ["assessment", "subgoals"]
    assert tuple(schema["$defs"]["SubgoalField"]["enum"]) == SUBGOAL_FIELDS
    assert schema["properties"]["subgoals"]["maxItems"] == 4


@pytest.mark.parametrize("field", SUBGOAL_FIELDS)
def test_every_subgoal_field_is_one_the_symbolic_planner_reads(field) -> None:
    from tiny_sima.planner.symbolic import PLANNER_FIELDS

    assert field in PLANNER_FIELDS


@pytest.mark.parametrize(
    ("chain_name", "answer"),
    [
        ("stone_pickaxe_terse", _answer(("inventory.stone_pickaxe", 1))),
        (
            "stone_pickaxe_and_sword",
            _answer(("inventory.wood_sword", 1), ("inventory.stone_pickaxe", 1)),
        ),
    ],
)
def test_closed_loop_reaches_the_objective_from_subgoals(chain_name, answer) -> None:
    from tiny_sima.evaluation.chains import (
        SKILL_CHAINS,
        ChainTermination,
        ScriptedSkillExecutor,
        chain_environment,
        chain_episode_record,
        evaluate_chain_episodes,
    )
    from tiny_sima.memory import EpisodeMemory

    chain = SKILL_CHAINS[chain_name]
    backend = ScriptedBackend(answer)
    planner = SubgoalPlanner(backend, candidate_skills=chain.skills)
    results = evaluate_chain_episodes(
        planner,
        ScriptedSkillExecutor(),
        chain_environment(chain),
        list(jax.random.split(jax.random.PRNGKey(7), 2)),
        chain=chain,
        memory_factory=EpisodeMemory,
    )

    for result in results:
        assert result.termination is ChainTermination.SUCCESS, result.termination
        # The plan holds after the first call: one VLM call per episode.
        assert result.trace[-1].plan_state.vlm_calls == 1
        record = json.loads(json.dumps(chain_episode_record(result)))
        assert record["trace"][0]["plan_state"]["trigger"] == "start"
    assert len(backend.requests) == 2
