import json
from typing import Any

from tiny_sima.evaluation.chains import (
    ScriptedSkillExecutor,
    evaluate_skill_chain,
    json_ready_chain_result,
)
from tiny_sima.evaluation.sequencing import (
    DecisionKind,
    choice_attribution,
    decision_kind,
    episode_sequence,
    off_objective_choices,
    planner_assessment,
    say_top_skill,
    sequencing_summary,
)
from tiny_sima.goals import SkillId
from tiny_sima.planner.saycan import SayCanPlanner, SequentialUtilityScorer


def _step(
    requested: str,
    *,
    outcome: str = "success",
    accepted: tuple[str, ...] = (),
    utilities: dict[str, float] | None = None,
    raw: str | None = None,
) -> dict[str, Any]:
    return {
        "requested_skill": requested,
        "outcome": outcome,
        "accepted_skills": list(accepted),
        "candidate_scores": [
            {"skill_id": skill, "utility": utility}
            for skill, utility in (utilities or {}).items()
        ],
        "planner_raw_response": raw,
    }


def _episode(*steps: dict[str, Any], termination: str = "decision_limit") -> dict:
    return {"termination": termination, "trace": list(steps)}


def test_decision_kind_reads_outcome_before_the_accepted_set() -> None:
    accepted = ("craft_wood_pickaxe",)

    assert decision_kind(_step("craft_wood_pickaxe", accepted=accepted)) is (
        DecisionKind.PROGRESS
    )
    assert decision_kind(_step("drink_water", accepted=accepted)) is (
        DecisionKind.OFF_OBJECTIVE
    )
    # An accepted skill the executor did not finish still bought nothing.
    assert (
        decision_kind(_step("craft_wood_pickaxe", outcome="timeout", accepted=accepted))
        is DecisionKind.FAILED
    )
    assert decision_kind(_step("drink_water")) is DecisionKind.UNKNOWN


def test_attribution_separates_the_say_term_from_the_can_term() -> None:
    accepted = ("craft_wood_pickaxe",)
    say_pickaxe = {"craft_wood_pickaxe": 1.0, "drink_water": 0.25}
    say_water = {"craft_wood_pickaxe": 0.25, "drink_water": 1.0}

    def attribute(requested: str, utilities: dict[str, float]) -> str | None:
        return choice_attribution(
            _step(requested, accepted=accepted, utilities=utilities)
        )

    assert attribute("craft_wood_pickaxe", say_pickaxe) == "both_accepted"
    assert attribute("craft_wood_pickaxe", say_water) == "can_rescued"
    assert attribute("drink_water", say_pickaxe) == "can_misled"
    assert attribute("drink_water", say_water) == "say_led"
    # A tie across the accepted set, as under constant_utility, expresses no
    # Say preference, whichever side the choice falls on.
    split_tie = {"craft_wood_pickaxe": 1.0, "drink_water": 1.0}
    assert say_top_skill(_step("drink_water", utilities=split_tie)) is None
    assert attribute("drink_water", split_tie) == "no_say_top"
    assert attribute("craft_wood_pickaxe", split_tie) == "no_say_top"
    # A tie on one side of it is still a preference.
    unaccepted_tie = {
        "drink_water": 1.0,
        "collect_wood": 1.0,
        "craft_wood_pickaxe": 0.5,
    }
    assert attribute("drink_water", unaccepted_tie) == "say_led"
    two_accepted = ("collect_wood", "place_table")
    accepted_tie = {"collect_wood": 1.0, "place_table": 1.0, "drink_water": 0.5}
    assert (
        choice_attribution(
            _step("drink_water", accepted=two_accepted, utilities=accepted_tie)
        )
        == "can_misled"
    )
    # The executor's outcome does not change whose choice it was.
    failed = _step(
        "craft_wood_pickaxe",
        outcome="timeout",
        accepted=accepted,
        utilities=say_pickaxe,
    )
    assert choice_attribution(failed) == "both_accepted"
    assert choice_attribution(_step("drink_water", utilities=say_water)) is None


def test_summary_counts_waste_repeats_and_patterns() -> None:
    accepted = ("craft_wood_pickaxe",)
    say_water = {"craft_wood_pickaxe": 0.25, "drink_water": 1.0}
    episodes = [
        _episode(
            _step("drink_water", accepted=accepted, utilities=say_water),
            _step("drink_water", accepted=accepted, utilities=say_water),
            _step(
                "craft_wood_pickaxe",
                outcome="timeout",
                accepted=accepted,
                utilities={"craft_wood_pickaxe": 1.0},
            ),
        ),
        _episode(
            _step("craft_wood_pickaxe", accepted=accepted, utilities=say_water),
            termination="success",
        ),
    ]

    summary = sequencing_summary(episodes)

    assert summary["num_episodes"] == 2
    assert summary["mean_decisions"] == 2.0
    assert summary["termination_counts"] == {"decision_limit": 1, "success": 1}
    assert summary["decision_kinds"] == {
        "progress": 1,
        "off_objective": 2,
        "failed": 1,
        "unknown": 0,
    }
    assert summary["attributions"] == {
        "both_accepted": 1,
        "can_rescued": 1,
        "can_misled": 0,
        "say_led": 2,
        "no_say_top": 0,
    }
    water = summary["per_skill"]["drink_water"]
    assert water["requests"] == 2
    assert water["repeats"] == 1
    assert water["off_objective"] == 2
    assert summary["per_skill"]["craft_wood_pickaxe"]["failed"] == 1
    assert summary["per_skill"]["collect_stone"]["requests"] == 0
    patterns = off_objective_choices(episodes)
    # The failed pickaxe is counted as failed, not as an off-objective pattern.
    assert list(patterns) == [(accepted, "drink_water", "drink_water")]
    assert (
        patterns[(accepted, "drink_water", "drink_water")] == episodes[0]["trace"][:2]
    )
    assert episode_sequence(episodes[0]) == "water* water* pickaxe!timeout"
    # The Say term's first choice is shown where the Can term overrode it.
    assert episode_sequence(episodes[1]) == "pickaxe(water)"
    assert sequencing_summary([])["num_episodes"] == 0


def test_off_objective_groups_come_most_frequent_first() -> None:
    accepted = ("collect_wood",)
    episodes = [
        _episode(
            _step("drink_water", accepted=accepted),
            _step("craft_wood_sword", accepted=accepted),
            _step("craft_wood_sword", accepted=accepted),
            _step("collect_wood", accepted=accepted),
        )
    ]

    patterns = off_objective_choices(episodes)

    assert [(chosen, len(steps)) for (_, _, chosen), steps in patterns.items()] == [
        ("craft_wood_sword", 2),
        ("drink_water", 1),
    ]


def test_planner_assessment_reads_the_raw_response() -> None:
    raw = json.dumps({"progress_assessment": "Need a pickaxe.", "ranking": []})

    assert planner_assessment(_step("drink_water", raw=raw)) == "Need a pickaxe."
    assert planner_assessment(_step("drink_water", raw="not json")) is None
    assert planner_assessment(_step("drink_water")) is None


def test_extra_wood_after_the_table_is_off_objective_in_a_real_chain() -> None:
    # Wood beyond the one the pickaxe costs, requested once the table stands.
    wasteful_plan = (
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.PLACE_TABLE,
        SkillId.COLLECT_WOOD,
        SkillId.COLLECT_WOOD,
        SkillId.CRAFT_WOOD_PICKAXE,
        SkillId.COLLECT_STONE,
    )
    result = evaluate_skill_chain(
        ScriptedSkillExecutor(),
        planner=SayCanPlanner(utility_scorer=SequentialUtilityScorer(wasteful_plan)),
        num_episodes=2,
        seed=7,
        skill_horizon=32,
    )
    episodes = json.loads(json.dumps(json_ready_chain_result(result)))["episodes"]

    assert result["success_rate"] == 1.0
    for episode in episodes:
        assert episode_sequence(episode) == "wood wood table wood wood* pickaxe stone"
    summary = sequencing_summary(episodes)
    assert summary["decision_kinds"]["off_objective"] == 2
    assert summary["decision_kinds"]["progress"] == 12
    # The scripted Say term put the extra wood first, so the waste is its own.
    assert summary["attributions"]["say_led"] == 2
    # Only the extra wood is a wasted repeat; the plan's opening pair is not.
    assert summary["per_skill"]["collect_wood"]["repeats"] == 2
