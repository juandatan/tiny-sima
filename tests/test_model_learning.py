import json

import jax

from tiny_sima.evaluation.chains import (
    WOOD_TO_STONE_PICKAXE_CHAIN,
    ScriptedSkillExecutor,
    chain_environment,
)
from tiny_sima.goals import SkillId, StateRequirement
from tiny_sima.memory import TransitionRecord
from tiny_sima.planner.model_learning import (
    analyse_failures,
    build_induction_prompt,
    correct_model,
    drop_invalid_skills,
    induce_model,
    model_diff,
    model_from_dict,
    model_to_dict,
    observed_effects,
    parse_failure_response,
    parse_induced_model,
    unexplained_failures,
    validate_model,
)
from tiny_sima.planner.model_rounds import (
    corrupt_registry,
    run_model_learning_rounds,
    subgoal_planner_factory,
)
from tiny_sima.planner.symbolic import REGISTRY_MODEL, shortest_plan

CHAIN = WOOD_TO_STONE_PICKAXE_CHAIN
GOAL = CHAIN.goal or ()
WOOD = "inventory.wood"


class FakeBackend:
    """Answer every request from a queue, the last answer repeating."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FormalGoalBackend:
    """Answer every sub-goal request with the chain's formal goal."""

    def __init__(self, goal):
        self.response = json.dumps(
            {
                "assessment": "formal goal",
                "subgoals": [
                    {"field": requirement.field, "at_least": int(requirement.value)}
                    for requirement in goal
                ],
            }
        )

    def generate(self, request):
        return self.response


def _record(
    skill_id, outcome="success", before=(), after=(), *, cause=None, extremes=None
):
    return TransitionRecord(
        decision_index=0,
        skill_id=skill_id,
        outcome=outcome,
        steps=4,
        before=(*before, ("survival.drink", 9)),
        after=(*after, ("survival.drink", 9)),
        cause=cause,
        inventory_extremes=extremes,
    )


def _mechanics(model, skill_id):
    return dict(model)[skill_id]


def _registry_json():
    return json.dumps(model_to_dict(REGISTRY_MODEL))


def test_registry_round_trips_and_validates():
    assert model_from_dict(model_to_dict(REGISTRY_MODEL)) == REGISTRY_MODEL
    assert validate_model(REGISTRY_MODEL).valid
    assert model_diff(REGISTRY_MODEL, REGISTRY_MODEL) == ()


def test_parse_drops_bad_entries_and_reports_them():
    payload = model_to_dict(REGISTRY_MODEL)
    payload["collect_wood"]["effects"].append(
        {"field": "inventory.mana", "operation": "increase", "value": 1}
    )
    payload["place_table"]["requires"].append(
        {"field": WOOD, "comparison": "at_least", "value": 3}
    )
    payload["craft_wood_sword"]["effects"][0]["value"] = 1.5
    del payload["drink_water"]
    parsed = parse_induced_model("```json\n" + json.dumps(payload) + "\n```")

    assert parsed.valid
    assert "drink_water: missing" in parsed.problems
    assert any("inventory.mana" in problem for problem in parsed.problems)
    assert any("repeated at_least" in problem for problem in parsed.problems)
    assert SkillId.DRINK_WATER not in dict(parsed.model)
    # The first requirement wins, and every skill gets its unlock.
    assert _mechanics(parsed.model, SkillId.PLACE_TABLE).requires[0].value == 2
    assert any(
        effect.field.startswith("achievement.")
        for effect in _mechanics(parsed.model, SkillId.COLLECT_WOOD).effects
    )
    assert not parse_induced_model("not json").valid
    assert not parse_induced_model("{}").valid


def test_validation_flags_each_rule_and_drop_cascades():
    payload = model_to_dict(REGISTRY_MODEL)
    # Only the wood pickaxe itself produces a wood pickaxe.
    payload["craft_wood_pickaxe"]["requires"].append(
        {"field": "inventory.wood_pickaxe", "comparison": "at_least", "value": 1}
    )
    payload["craft_wood_sword"]["effects"] = payload["collect_wood"]["effects"]
    payload["drink_water"]["effects"] = [
        {"field": "survival.drink", "operation": "increase", "value": 0}
    ]
    model = model_from_dict(payload)
    report = validate_model(model)

    rules = {(violation.skill_id, violation.rule) for violation in report.violations}
    assert rules == {
        (SkillId.DRINK_WATER, "no_effect"),
        (SkillId.CRAFT_WOOD_PICKAXE, "unreachable_requirements"),
        (SkillId.CRAFT_WOOD_SWORD, "duplicate_effects"),
        # Stone needs a wood pickaxe, which nothing can now make.
        (SkillId.COLLECT_STONE, "unreachable_requirements"),
        (SkillId.CRAFT_STONE_PICKAXE, "unreachable_requirements"),
    }
    kept, dropped = drop_invalid_skills(model)
    assert set(dropped) == {
        SkillId.DRINK_WATER,
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.CRAFT_WOOD_SWORD,
        SkillId.COLLECT_STONE,
        SkillId.CRAFT_STONE_PICKAXE,
    }
    assert validate_model(kept).valid


def test_induction_of_the_registry_plans_like_the_registry():
    backend = FakeBackend(_registry_json())
    induced = induce_model(backend, manual="A manual.")

    (request,) = backend.requests
    assert request.image is None
    assert request.output_type is not None
    request.output_type.model_validate_json(_registry_json())
    assert "inventory.wood" in build_induction_prompt()
    assert induced.valid and induced.report.valid
    assert induced.model == REGISTRY_MODEL

    state = chain_environment(CHAIN).reset(jax.random.PRNGKey(3)).state
    for counts in ({}, {"wood": 3}, {"wood": 1, "wood_pickaxe": 1, "stone": 1}):
        start = state.replace(inventory=state.inventory.replace(**counts))
        assert shortest_plan(start, GOAL, model=induced.model) == shortest_plan(
            start, GOAL, model=REGISTRY_MODEL
        )


def test_induction_repairs_an_invalid_model_once():
    payload = model_to_dict(REGISTRY_MODEL)
    payload["collect_wood"]["requires"].append(
        {"field": WOOD, "comparison": "at_least", "value": 1}
    )
    backend = FakeBackend(json.dumps(payload), _registry_json())
    induced = induce_model(backend, manual="A manual.", repair=True)

    assert len(backend.requests) == 2
    assert "collect_wood: unreachable requirements" in backend.requests[1].user_prompt
    assert induced.repaired and induced.model == REGISTRY_MODEL


def test_feasibility_names_the_root_cause_and_prompt_lists_starts():
    # As the 2B model once answered: the table needs a table and never
    # makes one, so nothing at a table is reachable either.
    payload = model_to_dict(REGISTRY_MODEL)
    payload["place_table"]["requires"].append(
        {
            "field": "world.crafting_table_reachable",
            "comparison": "equals",
            "value": 1,
        }
    )
    payload["place_table"]["effects"] = [
        {"field": WOOD, "operation": "decrease", "value": 2}
    ]
    report = validate_model(model_from_dict(payload))

    crafts = [
        str(violation)
        for violation in report.violations
        if violation.skill_id
        in {SkillId.PLACE_TABLE, SkillId.CRAFT_WOOD_PICKAXE, SkillId.CRAFT_WOOD_SWORD}
    ]
    assert len(crafts) == 3
    for reason in crafts:
        assert "world.crafting_table_reachable" in reason
        assert "no available skill changes it" in reason
        assert WOOD not in reason
    prompt = build_induction_prompt()
    assert "world.crafting_table_reachable: " in prompt
    assert "else 0; starts at 0" in prompt
    assert "survival.drink: " in prompt and "starts at 9" in prompt


def test_extremes_reveal_wood_gathered_and_spent_within_a_skill():
    # One wood held, a second gathered, both spent on the table.
    record = _record(
        SkillId.PLACE_TABLE,
        before=((WOOD, 1),),
        extremes=((WOOD, 0, 2),),
    )
    assert observed_effects(record) == {WOOD: ("decrease", 2)}
    model = corrupt_registry("table_costs_1_wood")
    corrected = correct_model(model, [record], min_support=1)

    table = _mechanics(corrected.model, SkillId.PLACE_TABLE)
    assert StateRequirement(WOOD, "at_least", 2) in table.requires
    assert (WOOD, "decrease", 2) in {
        (effect.field, effect.operation, effect.value) for effect in table.effects
    }
    assert {str(change) for change in corrected.changes} == {
        "place_table.requires: inventory.wood at_least 1 -> at_least 2",
        "place_table.effects: inventory.wood decrease 1 -> decrease 2",
    }


def test_correction_needs_support_for_a_new_requirement():
    # The stone was held in both successes, but the model never required it.
    records = [
        _record(
            SkillId.CRAFT_WOOD_PICKAXE,
            before=((WOOD, 2), ("inventory.stone", 1)),
            after=((WOOD, 1), ("inventory.stone", 1), ("inventory.wood_pickaxe", 1)),
        )
    ] * 2
    loose = correct_model(REGISTRY_MODEL, records, min_support=2)
    strict = correct_model(REGISTRY_MODEL, records, min_support=3)

    assert (
        StateRequirement("inventory.stone", "at_least", 1)
        in _mechanics(loose.model, SkillId.CRAFT_WOOD_PICKAXE).requires
    )
    assert strict.changes == ()


def _failure_answer(field, value=1):
    return json.dumps(
        {
            "analysis": "every failure lacked a pickaxe",
            "missing_requirements": [
                {"field": field, "comparison": "at_least", "value": value}
            ],
        }
    )


def test_failure_analysis_adds_the_missing_requirement():
    model = corrupt_registry("stone_without_pickaxe")
    pickaxe = ("inventory.wood_pickaxe", 1)
    records = [
        _record(SkillId.COLLECT_STONE, "infeasible", cause="infeasible"),
        _record(SkillId.COLLECT_STONE, "timeout", (pickaxe,), cause="target_not_seen"),
        _record(
            SkillId.COLLECT_STONE,
            before=(pickaxe,),
            after=(pickaxe, ("inventory.stone", 1)),
        ),
    ]
    unexplained = unexplained_failures(model, records)
    assert [len(found) for found in unexplained.values()] == [1]

    backend = FakeBackend(_failure_answer("inventory.wood_pickaxe"))
    learned, (analysis,) = analyse_failures(backend, model, records)
    assert analysis.applied
    assert set(dict(learned)[SkillId.COLLECT_STONE].requires) == set(
        dict(REGISTRY_MODEL)[SkillId.COLLECT_STONE].requires
    )
    (request,) = backend.requests
    assert "wood_pickaxe=1" in request.user_prompt

    # A requirement a success went without is refused.
    rejected, (refusal,) = analyse_failures(
        FakeBackend(_failure_answer("inventory.wood_pickaxe", 2)), model, records
    )
    assert not refusal.applied and rejected == model
    assert parse_failure_response(
        '{"analysis": "", "missing_requirements": []}', [WOOD]
    ) == (
        None,
        None,
    )


def test_failure_analysis_replaces_a_requirement_it_contradicts():
    # Induction inverted the pickaxe requirement; runs show stone needs one.
    payload = model_to_dict(REGISTRY_MODEL)
    payload["collect_stone"]["requires"][0]["comparison"] = "equals"
    payload["collect_stone"]["requires"][0]["value"] = 0
    model = model_from_dict(payload)
    pickaxe = ("inventory.wood_pickaxe", 1)
    records = [
        _record(SkillId.COLLECT_STONE, "infeasible", cause="infeasible"),
        _record(
            SkillId.COLLECT_STONE,
            before=(pickaxe,),
            after=(pickaxe, ("inventory.stone", 1)),
        ),
    ]
    backend = FakeBackend(_failure_answer("inventory.wood_pickaxe"))
    learned, (analysis,) = analyse_failures(backend, model, records)
    assert analysis.applied
    assert set(_mechanics(learned, SkillId.COLLECT_STONE).requires) == set(
        _mechanics(REGISTRY_MODEL, SkillId.COLLECT_STONE).requires
    )


def test_rounds_correct_a_cheap_table_back_to_the_registry():
    planners = subgoal_planner_factory(FormalGoalBackend(GOAL), CHAIN)
    result = run_model_learning_rounds(
        corrupt_registry("table_costs_1_wood"),
        planners,
        ScriptedSkillExecutor(),
        chain=CHAIN,
        rounds=2,
        episodes_per_round=2,
        seed=5,
    )
    json.dumps(result)

    first, second = result["rounds"]
    assert {change["skill"] for change in first["diff_vs_reference"]} == {"place_table"}
    assert not any(
        change["skill"] == "place_table" for change in second["diff_vs_reference"]
    )
    table = model_from_dict(result["final_model"])
    assert dict(table)[SkillId.PLACE_TABLE] == dict(REGISTRY_MODEL)[SkillId.PLACE_TABLE]
    assert second["success_rate"] == 1.0


def test_focused_questions_repair_what_a_whole_repair_repeats():
    # The 2B model's two mistakes: the table needs a table and never makes
    # one, and the wood pickaxe needs stone that needs the wood pickaxe.
    payload = model_to_dict(REGISTRY_MODEL)
    payload["place_table"]["requires"].append(
        {
            "field": "world.crafting_table_reachable",
            "comparison": "equals",
            "value": 1,
        }
    )
    payload["place_table"]["effects"] = [
        {"field": WOOD, "operation": "decrease", "value": 2}
    ]
    payload["craft_wood_pickaxe"]["requires"].append(
        {"field": "inventory.stone", "comparison": "at_least", "value": 1}
    )
    wrong = json.dumps(payload)
    backend = FakeBackend(
        wrong,
        wrong,
        json.dumps({"answer": "place_table"}),
        json.dumps({"answer": "yes"}),
        json.dumps({"answer": "no"}),
    )
    induced = induce_model(backend, manual="A manual.", repair=True, drop_invalid=True)

    questions = [request.user_prompt for request in backend.requests[2:]]
    assert questions == [
        "world.crafting_table_reachable is 1 while a crafting table is within "
        "reach, else 0. It is 0 at the start of every episode. Which skill "
        "makes it 1?",
        "Collect stone with a wood pickaxe: must the player hold a wood "
        "pickaxe first?",
        "Craft a wood pickaxe at a crafting table: must the player hold "
        "stone first?",
    ]
    assert [question.change for question in induced.questions] == [
        "place_table now brings about world.crafting_table_reachable == 1",
        "",
        "dropped: craft_wood_pickaxe needs inventory.stone >= 1",
    ]
    assert induced.report.valid and induced.dropped == ()
    assert induced.model == REGISTRY_MODEL
    json.dumps(induced.to_dict())


def test_absence_question_finds_the_deadlock_an_inverted_requirement_hides():
    # The GPU run's model: stone needs no pickaxe, the pickaxe needs stone.
    # It validates, so only the absence question exposes the deadlock.
    payload = model_to_dict(REGISTRY_MODEL)
    payload["collect_stone"]["requires"][0] = {
        "field": "inventory.wood_pickaxe",
        "comparison": "equals",
        "value": 0,
    }
    payload["craft_wood_pickaxe"]["requires"].append(
        {"field": "inventory.stone", "comparison": "at_least", "value": 1}
    )
    backend = FakeBackend(
        json.dumps(payload),
        json.dumps({"answer": "yes"}),
        json.dumps({"answer": "no"}),
    )
    induced = induce_model(backend, manual="A manual.", repair=True, drop_invalid=True)

    # The needless question about the pickaxe repeats the absence question,
    # so it is not asked again.
    assert [request.user_prompt for request in backend.requests[1:]] == [
        "Collect stone with a wood pickaxe: must the player hold a wood "
        "pickaxe first?",
        "Craft a wood pickaxe at a crafting table: must the player hold "
        "stone first?",
    ]
    assert [(question.kind, question.change) for question in induced.questions] == [
        (
            "absence",
            "collect_stone now needs inventory.wood_pickaxe >= 1, "
            "not inventory.wood_pickaxe == 0",
        ),
        ("needless", "dropped: craft_wood_pickaxe needs inventory.stone >= 1"),
    ]
    assert induced.report.valid and induced.dropped == ()
    assert induced.model == REGISTRY_MODEL

    # Answered no, the absence requirement is dropped instead.
    backend = FakeBackend(json.dumps(payload), json.dumps({"answer": "no"}))
    dropped = induce_model(backend, manual="A manual.", repair=True)
    assert _mechanics(dropped.model, SkillId.COLLECT_STONE).requires == (
        StateRequirement("world.stone_reachable", "equals", True),
    )
    assert dropped.report.valid
